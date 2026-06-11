"""
core/sia_attack.py
SIA — Structure Invariant Attack, ICCV 2023
Paper: arxiv.org/abs/2309.14700
Idea: split face image into 4x4 blocks, apply DIFFERENT random transform to
each block, average gradient over N=20 such diverse copies per step.
Far more diverse than global transforms (TI-FGSM, SIM) -> better transfer.
"""
import numpy as np
import tensorflow as tf


def cosine_sim_np(a, b):
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-10 else 0.0


def _preprocess(image_np, model_name):
    sizes = {
        "Facenet512":   (160, 160),
        "ArcFace":      (112, 112),
        "GhostFaceNet": (112, 112),
        "VGG-Face":     (224, 224),
    }
    h, w = sizes.get(model_name, (112, 112))
    img = tf.image.resize(image_np[np.newaxis], [h, w])
    img = tf.cast(img, tf.float32)
    img = (img - 127.5) / 128.0
    return img


def _load_keras_model(model_name):
    from deepface import DeepFace
    return DeepFace.build_model(model_name)


def _transform_block(block, tid):
    """Apply one of 10 transforms to a single image block (H,W,3)."""
    H, W = block.shape[:2]
    if   tid == 0: return block
    elif tid == 1: return block[:, ::-1, :]
    elif tid == 2:
        nh, nw = max(1, int(H*0.8)), max(1, int(W*0.8))
        small = tf.image.resize(block[np.newaxis],[nh,nw]).numpy()[0]
        out = np.zeros_like(block)
        ph, pw = (H-nh)//2, (W-nw)//2
        out[ph:ph+nh, pw:pw+nw] = small
        return out
    elif tid == 3:
        grey = np.mean(block, axis=2, keepdims=True)
        return np.broadcast_to(grey, block.shape).copy()
    elif tid == 4: return np.clip(block * 1.3 - 0.15, -1.0, 1.0)
    elif tid == 5: return np.clip(block + 0.1, -1.0, 1.0)
    elif tid == 6: return block[::-1, :, :]
    elif tid == 7: return np.rot90(block, k=1)
    elif tid == 8: return np.rot90(block, k=2)
    elif tid == 9: return np.rot90(block, k=3)
    return block


def structure_invariant_transform(image_tf, grid=4):
    """Split (1,H,W,C) tensor into grid*grid blocks, transform each randomly."""
    img_np = image_tf.numpy()[0]
    H, W, C = img_np.shape
    bH, bW  = H // grid, W // grid
    out = img_np.copy()
    for row in range(grid):
        for col in range(grid):
            r0 = row * bH
            r1 = (row+1)*bH if row < grid-1 else H
            c0 = col * bW
            c1 = (col+1)*bW if col < grid-1 else W
            block = img_np[r0:r1, c0:c1, :]
            tid   = np.random.randint(0, 10)
            t     = _transform_block(block, tid)
            if t.shape != block.shape:
                t = tf.image.resize(t[np.newaxis],[r1-r0,c1-c0]).numpy()[0]
            out[r0:r1, c0:c1, :] = t
    return tf.constant(out[np.newaxis], dtype=tf.float32)


def sia_attack(model_name, image_np, target_emb_np,
               attack_type="impersonation_attack",
               eps=0.05, steps=10, decay=1.0,
               num_copies=20, grid=4):
    """
    SIA transfer attack for face verification.

    model_name    : "Facenet512"|"ArcFace"|"GhostFaceNet"|"VGG-Face"
    image_np      : source face (H,W,3) float32 [0,1]
    target_emb_np : target embedding (D,) float32
    attack_type   : "impersonation_attack" or "dodging_attack"
    Returns       : adversarial image (H,W,3) float32 [0,1]
    """
    keras_model = _load_keras_model(model_name)
    alpha = eps / steps
    img_tf = _preprocess(image_np, model_name)
    tgt = tf.constant(target_emb_np.flatten(), dtype=tf.float32)
    tgt = tf.nn.l2_normalize(tgt, axis=0)

    adv      = tf.Variable(tf.identity(img_tf), trainable=True)
    momentum = tf.zeros_like(img_tf)

    for _ in range(steps):
        grad_accum = tf.zeros_like(adv)

        for _ in range(num_copies):
            xt  = structure_invariant_transform(adv, grid=grid)
            xv  = tf.Variable(xt, trainable=True)
            with tf.GradientTape() as tape:
                tape.watch(xv)
                emb  = keras_model(xv, training=False)
                emb_n = tf.nn.l2_normalize(tf.reshape(emb, [-1]), axis=0)
                cos  = tf.reduce_sum(emb_n * tgt)
                loss = cos if attack_type == "impersonation_attack" else (1.0-cos)
            g = tape.gradient(loss, xv)
            grad_accum = grad_accum + g / num_copies

        g_norm   = grad_accum / (tf.reduce_mean(tf.abs(grad_accum)) + 1e-8)
        momentum = decay * momentum + g_norm
        adv = adv + alpha * tf.sign(momentum)
        adv = tf.clip_by_value(adv, img_tf - eps, img_tf + eps)
        adv = tf.clip_by_value(adv, -1.0, 1.0)
        adv = tf.Variable(adv, trainable=True)

    out = adv.numpy()[0]
    out = (out * 128.0 + 127.5) / 255.0
    return np.clip(out, 0.0, 1.0).astype(np.float32)