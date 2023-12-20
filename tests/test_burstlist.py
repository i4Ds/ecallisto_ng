from ecallisto_ng.burst_list.utils import load_burst_list


def test_burst_list():
    burst_list = load_burst_list()
    assert not burst_list.empty
