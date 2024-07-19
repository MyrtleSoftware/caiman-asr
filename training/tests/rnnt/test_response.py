from caiman_asr_train.rnnt.response import DecodingResponse, HypothesisResponse


def test_response():
    hypothesis = HypothesisResponse(
        token_seq=["▁hello", "▁wo", "rld"],
        y_seq=[1, 2, 3],
        timesteps=[0, 1, 4],
        confidence=1.0,
    )

    assert hypothesis.y_seq == [1, 2, 3]
    assert hypothesis.timesteps == [0, 1, 4]
    assert hypothesis.token_seq == ["▁hello", "▁wo", "rld"]
    assert hypothesis.confidence == 1.0

    response = DecodingResponse(
        start_frame_idx=0,
        duration_frames=2,
        is_provisional=False,
        alternatives=[hypothesis],
    )

    assert response.start_frame_idx == 0
    assert response.duration_frames == 2
    assert response.is_provisional is False
    assert response.alternatives == [hypothesis]

    assert response.dict() == {
        "start_frame_idx": 0.0,
        "duration_frames": 2,
        "is_provisional": False,
        "alternatives": [
            {
                "token_seq": ["▁hello", "▁wo", "rld"],
                "y_seq": [1, 2, 3],
                "timesteps": [0, 1, 4],
                "confidence": 1.0,
            },
        ],
    }
