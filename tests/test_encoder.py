import torch
import pytest
from gridcast.core.encoder import SimpleEncoder

def test_encoder_decode_roundtrip():
    """Test that encode->decode preserves original grid."""
    encoder = SimpleEncoder(steps=4, width=3, height=3)
    
    original = torch.tensor([
        [1, 2, 1],
        [0, 1, 2],
        [2, 0, 1]
    ])
    
    encoded = encoder.encode(original)
    decoded = encoder.decode(encoded)
    
    assert torch.equal(decoded, original)
    assert encoded.shape == (4, 16, 3, 3)

if __name__ == "__main__":
    pytest.main([__file__])