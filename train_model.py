#!/usr/bin/env python3
"""
PlagaScan — train_model.py

Transfer learning con MobileNetV2 para el clasificador de 3 clases
(no_frijol / plaga / sano). Version final: dataset moderado y curado
(327 no_frijol), sin tecnicas de balanceo especiales -- confirmado en
experimentacion sistematica del 28 ago 2026 como el mejor punto de
balance entre variedad de no_frijol y precision en sano/plaga real.

Uso:
    python train_model.py --root ~/plagascan/dataset --epochs_head 10 --epochs_finetune 8
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

IMG_SIZE = 224
BATCH_SIZE = 32
SEED = 42


def build_datasets(dataset_root: Path):
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        seed=SEED,
        label_mode="int",
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        seed=SEED,
        label_mode="int",
        shuffle=False,
    )

    class_names = train_ds.class_names
    print(f"Clases detectadas: {class_names} (orden importa para el label)")

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=autotune)
    val_ds_eval = val_ds
    val_ds = val_ds.prefetch(buffer_size=autotune)

    return train_ds, val_ds, val_ds_eval, class_names


def build_model():
    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = preprocess(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(3, activation="softmax")(x)
    model = models.Model(inputs, outputs)

    return model, base_model


def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def plot_curves(history_head, history_finetune, out_path):
    acc = history_head.history["accuracy"] + (history_finetune.history["accuracy"] if history_finetune else [])
    val_acc = history_head.history["val_accuracy"] + (history_finetune.history["val_accuracy"] if history_finetune else [])
    loss = history_head.history["loss"] + (history_finetune.history["loss"] if history_finetune else [])
    val_loss = history_head.history["val_loss"] + (history_finetune.history["val_loss"] if history_finetune else [])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(acc, label="train")
    axes[0].plot(val_acc, label="val")
    if history_finetune:
        axes[0].axvline(len(history_head.history["accuracy"]) - 0.5, color="gray", linestyle="--", label="inicio fine-tuning")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Época")
    axes[0].legend()

    axes[1].plot(loss, label="train")
    axes[1].plot(val_loss, label="val")
    if history_finetune:
        axes[1].axvline(len(history_head.history["loss"]) - 0.5, color="gray", linestyle="--", label="inicio fine-tuning")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Época")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    print(f"Guardado: {out_path}")


def evaluate_and_report(model, val_ds_eval, class_names, out_dir: Path):
    from sklearn.metrics import classification_report, confusion_matrix
    import itertools

    y_true = np.concatenate([y.numpy() for _, y in val_ds_eval], axis=0).flatten()
    y_pred_prob = model.predict(val_ds_eval)
    y_pred = np.argmax(y_pred_prob, axis=1)

    report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
    print("\n=== REPORTE DE CLASIFICACIÓN (val/) ===")
    print(report)
    (out_dir / "classification_report.txt").write_text(report)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusión (val/)")
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=130)
    print(f"Guardado: {out_dir / 'confusion_matrix.png'}")


def convert_to_tflite(model, out_path: Path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    out_path.write_bytes(tflite_model)
    size_kb = out_path.stat().st_size / 1024
    print(f"Guardado: {out_path} ({size_kb:.0f} KB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="~/plagascan/dataset")
    parser.add_argument("--out", type=str, default="~/plagascan/model")
    parser.add_argument("--epochs_head", type=int, default=10)
    parser.add_argument("--epochs_finetune", type=int, default=8)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--lr_finetune", type=float, default=1e-5)
    parser.add_argument("--finetune_layers", type=int, default=30)
    args = parser.parse_args()

    dataset_root = Path(args.root).expanduser()
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPUs detectadas: {tf.config.list_physical_devices('GPU')}")

    train_ds, val_ds, val_ds_eval, class_names = build_datasets(dataset_root)

    model, base_model = build_model()
    compile_model(model, args.lr_head)
    model.summary()

    print("\n=== FASE 1: entrenando solo la cabeza (MobileNetV2 congelado) ===")
    history_head = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_head)

    history_finetune = None
    if args.epochs_finetune > 0:
        print(f"\n=== FASE 2: fine-tuning (descongelando últimas {args.finetune_layers} capas) ===")
        base_model.trainable = True
        for layer in base_model.layers[:-args.finetune_layers]:
            layer.trainable = False
        compile_model(model, args.lr_finetune)
        history_finetune = model.fit(
            train_ds, validation_data=val_ds,
            epochs=args.epochs_head + args.epochs_finetune,
            initial_epoch=args.epochs_head,
        )

    model.save(out_dir / "plagascan_mobilenetv2.keras")
    print(f"Guardado: {out_dir / 'plagascan_mobilenetv2.keras'}")

    plot_curves(history_head, history_finetune, out_dir / "training_curves.png")
    evaluate_and_report(model, val_ds_eval, class_names, out_dir)
    convert_to_tflite(model, out_dir / "plagascan_mobilenetv2.tflite")

    print("\n=== ENTRENAMIENTO COMPLETO ===")
    print(f"Todos los archivos en: {out_dir}")


if __name__ == "__main__":
    main()
