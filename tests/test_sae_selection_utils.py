import pytest
from sae_bench_utils.sae_selection_utils import all_loadable_saes, get_saes_from_regex, print_all_sae_releases, print_release_details
from unittest.mock import patch, MagicMock

@pytest.fixture
def mock_pretrained_saes_directory():
    mock_directory = {
        'release1': MagicMock(
            saes_map={'sae1': 'path1', 'sae2': 'path2'},
            expected_var_explained={'sae1': 0.9, 'sae2': 0.8},
            expected_l0={'sae1': 10, 'sae2': 20},
        ),
        'release2': MagicMock(
            saes_map={'sae3': 'path3', 'sae4': 'path4'},
            expected_var_explained={'sae3': 0.7, 'sae4': 0.6},
            expected_l0={'sae3': 30, 'sae4': 40},
        ),
    }
    return mock_directory

def test_all_loadable_saes(mock_pretrained_saes_directory):
    with patch('sae_bench_utils.sae_selection_utils.get_pretrained_saes_directory', return_value=mock_pretrained_saes_directory):
        result = all_loadable_saes()
        assert len(result) == 4
        assert ('release1', 'sae1', 0.9, 10) in result
        assert ('release1', 'sae2', 0.8, 20) in result
        assert ('release2', 'sae3', 0.7, 30) in result
        assert ('release2', 'sae4', 0.6, 40) in result

def test_get_saes_from_regex(mock_pretrained_saes_directory):
    with patch('sae_bench_utils.sae_selection_utils.get_pretrained_saes_directory', return_value=mock_pretrained_saes_directory):
        result = get_saes_from_regex(r"release1", r"sae\d")
        assert result == {'release1': ['sae1', 'sae2']}

        result = get_saes_from_regex(r"release2", r"sae3")
        assert result == {'release2': ['sae3']}

        result = get_saes_from_regex(r"release\d", r"sae[24]")
        assert result == {'release1': ['sae2'], 'release2': ['sae4']}


def test_print_all_sae_releases(capsys):
    mock_directory = {
        'release1': MagicMock(
            model='model1',
            release='release1',
            repo_id='repo1',
            saes_map={'sae1': 'path1', 'sae2': 'path2'}
        ),
        'release2': MagicMock(
            model='model2',
            release='release2',
            repo_id='repo2',
            saes_map={'sae3': 'path3', 'sae4': 'path4'}
        ),
    }

    with patch('sae_bench_utils.sae_selection_utils.get_pretrained_saes_directory', return_value=mock_directory):
        print_all_sae_releases()
        captured = capsys.readouterr()
        
        # Check if the output contains the expected information
        assert "model1" in captured.out
        assert "model2" in captured.out
        assert "release1" in captured.out
        assert "release2" in captured.out
        assert "repo1" in captured.out
        assert "repo2" in captured.out
        assert "2" in captured.out  # number of SAEs for each release

def test_print_release_details(capsys):
    mock_release = MagicMock(
        model='model1',
        release='release1',
        repo_id='repo1',
        saes_map={'sae1': 'path1', 'sae2': 'path2'},
        expected_var_explained={'sae1': 0.9, 'sae2': 0.8},
        expected_l0={'sae1': 10, 'sae2': 20},
    )
    mock_directory = {'release1': mock_release}

    with patch('sae_bench_utils.sae_selection_utils.get_pretrained_saes_directory', return_value=mock_directory):
        print_release_details('release1')
        captured = capsys.readouterr()
        
        # Check if the output contains the expected information
        assert "release1" in captured.out
        assert "model1" in captured.out
        assert "repo1" in captured.out
        assert "saes_map" in captured.out
        assert "expected_var_explained" in captured.out
        assert "expected_l0" in captured.out