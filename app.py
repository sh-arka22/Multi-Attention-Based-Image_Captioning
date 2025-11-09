import argparse
import numpy as np
import pickle
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from functools import lru_cache
from tensorflow.keras.layers import (
    Input, Embedding, Dropout, Dense, LSTM, Bidirectional, RepeatVector,
    AdditiveAttention, Attention, MultiHeadAttention, concatenate, Lambda
)

# Keras 3 serialization helper (try keras first, then tf.keras as fallback)
try:
    from keras.saving import register_keras_serializable
except ImportError:
    from tensorflow.keras.utils import register_keras_serializable


# =========================
# Compatibility LSTM (ignores time_major)
# =========================
@register_keras_serializable(name="LegacyLSTM")
class LegacyLSTM(tf.keras.layers.LSTM):
    """
    Compatibility LSTM that accepts legacy `time_major` from old configs
    and just ignores it.
    """

    def __init__(self, *args, time_major=False, **kwargs):
        # Swallow `time_major` and forward everything else to real LSTM
        super().__init__(*args, **kwargs)


# =========================
# Custom AttentionBlock
# =========================
class AttentionBlock(tf.keras.layers.Layer):
    """
    Apply attention between text_seq (queries) and image_seq (keys/values),
    then masked-mean pool over time to get a (B, D) context vector.

    kind: 'luong' | 'bahdanau' | 'scaled_dot' | 'multihead' | 'concatenate'
    """
    def __init__(self, kind="luong", num_heads=4, dropout=0.3, scale=False, **kwargs):
        super().__init__(**kwargs)
        self.kind = kind
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.scale = scale

        if kind == "bahdanau":
            self.attn = AdditiveAttention(use_scale=True, dropout=dropout)
        elif kind == "scaled_dot":
            self.attn = Attention(use_scale=True, dropout=dropout)
        elif kind == "multihead":
            self.mha = MultiHeadAttention(
                num_heads=num_heads,
                key_dim=64,
                output_shape=512,
                dropout=dropout
            )
        elif kind == "concatenate":
            self.fuse = Dense(512, activation='tanh')
            self.fuse_do = Dropout(dropout)
        elif kind == "luong":
            self.do = Dropout(dropout)
        else:
            raise ValueError(f"Unknown attention kind: {kind}")

    def call(self, inputs):
        image_seq, text_seq, tokens = inputs          # (B,T,512), (B,T,512), (B,T)
        text_mask = tf.not_equal(tokens, 0)           # (B,T) boolean
        mask_f = tf.cast(text_mask, tf.float32)[..., None]  # (B,T,1)

        if self.kind == "bahdanau":
            attended = self.attn(
                [text_seq, image_seq, image_seq],
                mask=[text_mask, None]
            )  # (B,T,512)

        elif self.kind == "scaled_dot":
            attended = self.attn(
                [text_seq, image_seq],
                mask=[text_mask, None]
            )  # (B,T,512)

        elif self.kind == "multihead":
            Tq = tf.shape(text_seq)[1]
            Tk = tf.shape(image_seq)[1]
            attn_mask = tf.cast(tf.expand_dims(text_mask, -1), tf.int32)  # (B,T,1)
            attn_mask = tf.tile(attn_mask, [1, 1, Tk])                    # (B,T,T)
            attended = self.mha(
                query=text_seq,
                key=image_seq,
                value=image_seq,
                attention_mask=attn_mask
            )  # (B,T,512)

        elif self.kind == "concatenate":
            fused = tf.concat([text_seq, image_seq], axis=-1)   # (B,T,1024)
            fused = self.fuse(fused)                            # (B,T,512)
            attended = self.fuse_do(fused)                      # (B,T,512)

        else:  # 'luong'
            scores = tf.matmul(text_seq, image_seq, transpose_b=True)  # (B,T,T)
            if self.scale:
                d = tf.cast(tf.shape(text_seq)[-1], tf.float32)
                scores = scores / tf.sqrt(d)
            weights = tf.nn.softmax(scores, axis=-1)                   # (B,T,T)
            weights = self.do(weights)
            attended = tf.matmul(weights, image_seq)                   # (B,T,512)

        attended_sum = tf.reduce_sum(attended * mask_f, axis=1)
        denom = tf.reduce_sum(mask_f, axis=1) + 1e-9
        context = attended_sum / denom
        return context

    def get_config(self):
        base = super().get_config()
        base.update(
            dict(
                kind=self.kind,
                num_heads=self.num_heads,
                dropout=self.dropout_rate,
                scale=self.scale
            )
        )
        return base


# Optional helper (in case it's referenced in the model)
def last_valid_timestep():
    def _fn(inputs):
        x, tok = inputs            # x: (B, T, D), tok: (B, T), 0 = pad
        mask = tf.not_equal(tok, 0)
        lengths = tf.reduce_sum(tf.cast(mask, tf.int32), axis=1)   # (B,)
        idx = tf.maximum(lengths - 1, 0)
        b = tf.range(tf.shape(x)[0])
        return tf.gather_nd(x, tf.stack([b, idx], axis=1))         # (B, D)
    return Lambda(_fn, name="take_last_valid")


# =========================
# Paths
# =========================
DEFAULT_MODEL_PATH = Path("results/mymodel.h5")
DEFAULT_TOKENIZER_PATH = Path("results/tokenizer.pkl")


# =========================
# Loaders (with simple caching)
# =========================
@lru_cache(maxsize=1)
def load_feature_extractor():
    # VGG16 fc2 (4096-dim) to match training
    base_model = VGG16(weights="imagenet")
    return Model(
        inputs=base_model.inputs,
        outputs=base_model.layers[-2].output   # 'fc2' layer
    )


@lru_cache(maxsize=8)
def load_caption_model(model_path: str = str(DEFAULT_MODEL_PATH)):
    # Map both 'LSTM' and 'LegacyLSTM' to our compatibility class
    custom_objects = {
        "AttentionBlock": AttentionBlock,
        "LegacyLSTM": LegacyLSTM,
        "LSTM": LegacyLSTM,
        "Bidirectional": Bidirectional,
        "AdditiveAttention": AdditiveAttention,
        "Attention": Attention,
        "MultiHeadAttention": MultiHeadAttention,
    }

    model = tf.keras.models.load_model(
        model_path,
        custom_objects=custom_objects,
        compile=False,
        safe_mode=False,   # relax strictness for legacy H5 config
    )

    # input shapes: [(None,4096), (None,max_len)]
    text_input_shape = model.input_shape[1]
    max_length = int(text_input_shape[1])
    return model, max_length


@lru_cache(maxsize=8)
def load_tokenizer(tokenizer_path: str = str(DEFAULT_TOKENIZER_PATH)):
    with open(tokenizer_path, "rb") as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)
    index_word = getattr(tokenizer, "index_word", None)
    if not index_word:
        index_word = {idx: word for word, idx in tokenizer.word_index.items()}
    return tokenizer, index_word


# =========================
# Caption prediction
# =========================
def predict_caption(model, image_features, tokenizer, index_word, max_caption_length):
    """
    Greedy decoding:
    - start with 'startseq'
    - repeatedly predict next word
    - stop on 'endseq' or when max length reached
    """
    in_text = "startseq"

    for _ in range(max_caption_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_caption_length, padding="post")
        yhat = model.predict([image_features, seq], verbose=0)
        yhat = int(np.argmax(yhat))
        word = index_word.get(yhat)
        if word is None:
            break
        in_text += " " + word
        if word == "endseq":
            break

    tokens = [w for w in in_text.split() if w not in ("startseq", "endseq")]
    return " ".join(tokens)


# =========================
# Helper function to process image bytes
# =========================
def process_image_from_bytes(image_bytes, target_size=(224, 224)):
    """
    Process image bytes and return preprocessed array.
    
    Args:
        image_bytes: Image file bytes or file-like object
        target_size: Target image size (width, height)
    
    Returns:
        Preprocessed image array ready for VGG16
    """
    from io import BytesIO
    if isinstance(image_bytes, bytes):
        image_bytes = BytesIO(image_bytes)
    
    image = load_img(image_bytes, target_size=target_size)
    image_array = img_to_array(image)
    image_batch = np.expand_dims(image_array, axis=0)
    preprocessed_image = preprocess_input(image_batch)
    return preprocessed_image


# =========================
# Simple CLI helper (optional)
# =========================
def _parse_args():
    parser = argparse.ArgumentParser(description="Generate an image caption locally.")
    parser.add_argument("--image", type=Path, required=True, help="Path to input image.")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the trained caption model (.h5).",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=DEFAULT_TOKENIZER_PATH,
        help="Path to the tokenizer pickle.",
    )
    return parser.parse_args()


def _generate_caption_from_path(image_path: Path, model_path: Path, tokenizer_path: Path):
    import caption_core as core

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

    feature_extractor = core.load_feature_extractor()
    caption_model, max_caption_length = core.load_caption_model(str(model_path))
    tokenizer, index_word = core.load_tokenizer(str(tokenizer_path))

    with image_path.open("rb") as image_file:
        image_bytes = image_file.read()

    preprocessed_image = core.process_image_from_bytes(image_bytes)
    image_features = feature_extractor.predict(preprocessed_image, verbose=0)

    return core.predict_caption(
        caption_model,
        image_features,
        tokenizer,
        index_word,
        max_caption_length,
    )


if __name__ == "__main__":
    args = _parse_args()
    caption = _generate_caption_from_path(args.image, args.model, args.tokenizer)
    print(f'Caption: "{caption or "Caption not available"}"')
