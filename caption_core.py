import numpy as np
import pickle
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from functools import lru_cache
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Dropout,
    Dense,
    LSTM,
    Bidirectional,
    RepeatVector,
    AdditiveAttention,
    Attention,
    MultiHeadAttention,
    concatenate,
    Lambda,
)

# Keras 3 serialization helper (try keras first, then tf.keras as fallback)
try:
    from keras.saving import register_keras_serializable
except ImportError:
    from tensorflow.keras.utils import register_keras_serializable


@register_keras_serializable(name="LegacyLSTM")
class LegacyLSTM(tf.keras.layers.LSTM):
    """Compatibility LSTM that accepts legacy `time_major` from old configs."""

    def __init__(self, *args, time_major=False, **kwargs):
        super().__init__(*args, **kwargs)


class AttentionBlock(tf.keras.layers.Layer):
    """
    Apply attention between text_seq (queries) and image_seq (keys/values),
    then masked-mean pool over time to get a (B, D) context vector.
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
                num_heads=num_heads, key_dim=64, output_shape=512, dropout=dropout
            )
        elif kind == "concatenate":
            self.fuse = Dense(512, activation="tanh")
            self.fuse_do = Dropout(dropout)
        elif kind == "luong":
            self.do = Dropout(dropout)
        else:
            raise ValueError(f"Unknown attention kind: {kind}")

    def call(self, inputs):
        image_seq, text_seq, tokens = inputs  # (B,T,512), (B,T,512), (B,T)
        text_mask = tf.not_equal(tokens, 0)  # (B,T) boolean
        mask_f = tf.cast(text_mask, tf.float32)[..., None]  # (B,T,1)

        if self.kind == "bahdanau":
            attended = self.attn(
                [text_seq, image_seq, image_seq], mask=[text_mask, None]
            )
        elif self.kind == "scaled_dot":
            attended = self.attn([text_seq, image_seq], mask=[text_mask, None])
        elif self.kind == "multihead":
            Tk = tf.shape(image_seq)[1]
            attn_mask = tf.cast(tf.expand_dims(text_mask, -1), tf.int32)
            attn_mask = tf.tile(attn_mask, [1, 1, Tk])
            attended = self.mha(
                query=text_seq,
                key=image_seq,
                value=image_seq,
                attention_mask=attn_mask,
            )
        elif self.kind == "concatenate":
            fused = tf.concat([text_seq, image_seq], axis=-1)
            fused = self.fuse(fused)
            attended = self.fuse_do(fused)
        else:  # 'luong'
            scores = tf.matmul(text_seq, image_seq, transpose_b=True)
            if self.scale:
                d = tf.cast(tf.shape(text_seq)[-1], tf.float32)
                scores = scores / tf.sqrt(d)
            weights = tf.nn.softmax(scores, axis=-1)
            weights = self.do(weights)
            attended = tf.matmul(weights, image_seq)

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
                scale=self.scale,
            )
        )
        return base


def last_valid_timestep():
    def _fn(inputs):
        x, tok = inputs  # x: (B, T, D), tok: (B, T), 0 = pad
        mask = tf.not_equal(tok, 0)
        lengths = tf.reduce_sum(tf.cast(mask, tf.int32), axis=1)
        idx = tf.maximum(lengths - 1, 0)
        b = tf.range(tf.shape(x)[0])
        return tf.gather_nd(x, tf.stack([b, idx], axis=1))

    return Lambda(_fn, name="take_last_valid")


DEFAULT_MODEL_PATH = Path("results/mymodel.h5")
DEFAULT_TOKENIZER_PATH = Path("results/tokenizer.pkl")


@lru_cache(maxsize=1)
def load_feature_extractor():
    base_model = VGG16(weights="imagenet")
    return Model(
        inputs=base_model.inputs,
        outputs=base_model.layers[-2].output,
    )


@lru_cache(maxsize=8)
def load_caption_model(model_path: str = str(DEFAULT_MODEL_PATH)):
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
        safe_mode=False,
    )

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


def predict_caption(model, image_features, tokenizer, index_word, max_caption_length):
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


def process_image_from_bytes(image_bytes, target_size=(224, 224)):
    """Process image bytes and return preprocessed array."""

    from io import BytesIO

    if isinstance(image_bytes, bytes):
        image_bytes = BytesIO(image_bytes)

    image = load_img(image_bytes, target_size=target_size)
    image_array = img_to_array(image)
    image_batch = np.expand_dims(image_array, axis=0)
    preprocessed_image = preprocess_input(image_batch)
    return preprocessed_image


def build_image_caption_model(
    vocab_size: int,
    max_caption_length: int,
    attention_kind: str,
    learning_rate: float,
) -> Model:
    """
    Build an image caption model with specified attention mechanism.
    
    Args:
        vocab_size: Size of vocabulary
        max_caption_length: Maximum caption length
        attention_kind: Type of attention ("luong", "bahdanau", "scaled_dot", "multihead", "concatenate")
        learning_rate: Learning rate for optimizer
    
    Returns:
        Compiled Keras model
    """
    import re
    
    def safe_attention_name(kind: str) -> str:
        cleaned = re.sub(r"[^a-z0-9_]+", "_", kind.lower())
        cleaned = cleaned.strip("_")
        return cleaned or "attention"
    
    safe_name = safe_attention_name(attention_kind)

    inputs_image = Input(shape=(4096,), name="image")
    fe1 = Dropout(0.5)(inputs_image)
    fe2 = Dense(256, activation="relu", name="image_vec")(fe1)
    fe_seq = RepeatVector(max_caption_length, name="repeat_img")(fe2)
    fe_seq = Bidirectional(LSTM(256, return_sequences=True), name="img_seq_bilstm")(fe_seq)

    inputs_seq = Input(shape=(max_caption_length,), name="text")
    se1 = Embedding(vocab_size, 256, mask_zero=True, name="text_emb")(inputs_seq)
    se2 = Dropout(0.5)(se1)
    se3 = Bidirectional(LSTM(256, return_sequences=True), name="text_seq_bilstm")(se2)

    context = AttentionBlock(kind=attention_kind, num_heads=4, dropout=0.3, name=f"attn_{safe_name}")(
        [fe_seq, se3, inputs_seq]
    )

    decoder_input = concatenate([context, fe2], axis=-1)
    decoder_dense = Dense(256, activation="relu")(decoder_input)
    outputs = Dense(vocab_size, activation="softmax")(decoder_dense)

    model = Model(inputs=[inputs_image, inputs_seq], outputs=outputs, name=f"caption_{safe_name}")
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer)
    return model
