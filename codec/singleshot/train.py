"""Legacy singleshot training entrypoint.

Use train_singleshot.py for the current SwiftDecoder/MultiLevelAutoencoder
training flow. This file is kept as a compatibility shim so accidental runs do
not fail with undefined legacy symbols.
"""

try:
    from .train_singleshot import train_singleshot
except ImportError:
    from train_singleshot import train_singleshot


if __name__ == "__main__":
    train_singleshot()
