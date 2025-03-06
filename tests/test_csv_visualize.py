from lamoom_cicd import TestLLMResponsePipe
import os
import pytest
import dotenv
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import pandas as pd

dotenv.load_dotenv(dotenv.find_dotenv())


def test_csv_compare():
    lamoom_pipe = TestLLMResponsePipe(threshold=70, openai_key=os.environ.get("OPENAI_KEY"))
    results = lamoom_pipe.compare_from_csv("tests/test_data.csv")

    lamoom_pipe.visualize_test_results()

    assert 'score' in results[0].score.to_dict()
    assert 'passed' in results[0].score.to_dict()


def test_visualization_options():
    """Test the different visualization options."""
    # Create a temporary file for saving visualizations
    temp_dir = tempfile.mkdtemp()
    
    # Initialize the pipe
    lamoom_pipe = TestLLMResponsePipe(threshold=70, openai_key=os.environ.get("OPENAI_KEY"))
    
    # Create some test results with different prompt_ids and ideal_answers
    ideal_answers = [
        "The sky is blue because of Rayleigh scattering.",
        "Water boils at 100 degrees Celsius at sea level.",
        "The Earth orbits the Sun in an elliptical path."
    ]
    
    llm_responses = [
        "The sky appears blue due to a phenomenon called Rayleigh scattering.",
        "Water will boil at 100 degrees Celsius when at sea level pressure.",
        "The Earth moves around the Sun in an elliptical orbit."
    ]
    
    prompt_ids = ["prompt_v1", "prompt_v2"]
    
    # Create test results manually
    for i, ideal_answer in enumerate(ideal_answers):
        for j, prompt_id in enumerate(prompt_ids):
            optional_params = {"prompt_id": prompt_id, "prompt_version": f"version_{j+1}"}
            lamoom_pipe.compare(
                ideal_answer=ideal_answer,
                llm_response=llm_responses[i],
                optional_params=optional_params
            )
    
    # Test different visualization options
    
    # 1. Basic visualization by prompt_id (the default)
    save_path = os.path.join(temp_dir, "viz_by_prompt.png")
    lamoom_pipe.visualize_test_results(
        group_by="optional_param__prompt_id",
        save_path=save_path
    )
    assert os.path.exists(save_path)
    
    # 2. Visualization by ideal_answer
    save_path = os.path.join(temp_dir, "viz_by_ideal.png")
    lamoom_pipe.visualize_test_results(
        group_by="ideal_answer",
        save_path=save_path
    )
    assert os.path.exists(save_path)
    
    # 3. Visualization with statistics
    save_path = os.path.join(temp_dir, "viz_with_stats.png")
    lamoom_pipe.visualize_test_results(
        show_statistics=True,
        save_path=save_path
    )
    assert os.path.exists(save_path)
    
    # 4. Statistics visualization - bar chart
    save_path = os.path.join(temp_dir, "stats_bar.png")
    lamoom_pipe.visualize_statistics(
        visualization_type="bar",
        save_path=save_path
    )
    assert os.path.exists(save_path)
    
    # 5. Statistics visualization - box plot
    save_path = os.path.join(temp_dir, "stats_box.png")
    lamoom_pipe.visualize_statistics(
        visualization_type="box",
        save_path=save_path
    )
    assert os.path.exists(save_path)
    
    # 6. Statistics visualization - violin plot
    save_path = os.path.join(temp_dir, "stats_violin.png")
    lamoom_pipe.visualize_statistics(
        visualization_type="violin",
        save_path=save_path
    )
    assert os.path.exists(save_path)


def test_import_export_functionality():
    """Test the import and export functionality with version tracking."""
    # Create temporary files for exporting and importing
    temp_dir = tempfile.mkdtemp()
    export_path = os.path.join(temp_dir, "test_results.csv")
    
    # Create initial pipe
    pipe1 = TestLLMResponsePipe(threshold=70, openai_key=os.environ.get("OPENAI_KEY"))
    
    # Add some test results with 'v1' prompt
    ideal_answer = "The sky is blue because of Rayleigh scattering."
    llm_response = "The sky appears blue due to a phenomenon called Rayleigh scattering."
    
    pipe1.compare(
        ideal_answer=ideal_answer,
        llm_response=llm_response,
        optional_params={"prompt_id": "sky_color", "prompt_version": "v1"}
    )
    
    # Export results to CSV
    pipe1.export_results(export_path)
    assert os.path.exists(export_path)
    
    # Create a new pipe for 'v2' prompt tests
    pipe2 = TestLLMResponsePipe(threshold=70, openai_key=os.environ.get("OPENAI_KEY"))
    
    # Add results with 'v2' prompt
    pipe2.compare(
        ideal_answer=ideal_answer,
        llm_response="The blue color of the sky is caused by Rayleigh scattering of sunlight.",
        optional_params={"prompt_id": "sky_color", "prompt_version": "v2"}
    )
    
    # Import previous results and compare with new version
    pipe2.import_results(
        export_path, 
        label="v1",
        update_optional_params={"imported": True}
    )
    
    # Both pipes should have results
    assert len(pipe1.accumulated_results) == 1
    assert len(pipe2.accumulated_results) == 2
    
    # Compare results and verify they were imported correctly
    imported_result = next(r for r in pipe2.accumulated_results if r.prompt_id == "sky_color_v1")
    assert imported_result is not None
    assert imported_result.optional_params.get("imported") is True
    
    # Visualize the comparison
    save_path = os.path.join(temp_dir, "version_comparison.png")
    pipe2.visualize_test_results(save_path=save_path)
    assert os.path.exists(save_path)
    
    # Get statistics comparing the versions
    stats = pipe2.get_statistics()
    assert "sky_color" in stats or "sky_color_v2" in stats
    assert "sky_color_v1" in stats


def test_model_comparison():
    """Test comparing results across different models."""
    # Create temporary file for saving visualization
    temp_dir = tempfile.mkdtemp()
    save_path = os.path.join(temp_dir, "model_comparison.png")
    
    # Initialize the pipe
    lamoom_pipe = TestLLMResponsePipe(threshold=70, openai_key=os.environ.get("OPENAI_KEY"))
    
    # Create test data with different model responses
    ideal_answer = "The sky is blue because of Rayleigh scattering of sunlight."
    
    # Simulated responses from different models
    model_responses = {
        "gpt-4": "The sky appears blue due to a phenomenon called Rayleigh scattering, where sunlight is scattered by molecules in the atmosphere.",
        "claude-3": "Blue skies result from Rayleigh scattering, a process where air molecules scatter shorter wavelengths of sunlight more effectively.",
        "llama-3": "The sky is blue because the atmosphere scatters blue light from the sun more than red light. This is called Rayleigh scattering."
    }
    
    # Add test results for each model
    for model_name, response in model_responses.items():
        lamoom_pipe.compare(
            ideal_answer=ideal_answer,
            llm_response=response,
            optional_params={"model_name": model_name, "prompt_id": "sky_color"}
        )
    
    # Verify we have results for all models
    assert len(lamoom_pipe.accumulated_results) == 3
    
    # Visualize comparison grouped by model name
    lamoom_pipe.visualize_test_results(
        group_by="optional_param__model_name",
        show_statistics=True,
        save_path=save_path
    )
    assert os.path.exists(save_path)
    
    # Create a statistical visualization
    stats_path = os.path.join(temp_dir, "model_stats.png")
    lamoom_pipe.visualize_statistics(
        visualization_type="bar",
        save_path=stats_path
    )
    assert os.path.exists(stats_path)


def test_multi_dimension_visualization():
    """Test visualization with both group_by and group_line_by parameters."""
    # Create temporary file for saving visualization
    temp_dir = tempfile.mkdtemp()
    save_path = os.path.join(temp_dir, "multi_dimension.png")
    
    # Initialize the pipe
    lamoom_pipe = TestLLMResponsePipe(threshold=70, openai_key=os.environ.get("OPENAI_KEY"))
    
    # Create test data with different models, prompts and topics
    topics = [
        "The sky is blue because of Rayleigh scattering.",
        "Water boils at 100 degrees Celsius at sea level.",
        "The Earth orbits the Sun in an elliptical path."
    ]
    
    models = ["gpt-4", "claude-3", "llama-3"]
    prompt_versions = ["v1", "v2"]
    
    # Create a comprehensive dataset with variations across all dimensions
    for topic_idx, topic in enumerate(topics):
        for model in models:
            for version in prompt_versions:
                # Generate a slightly different response for each combination
                response = f"Model {model} using prompt {version} explains: {topic}"
                
                lamoom_pipe.compare(
                    ideal_answer=topic,
                    llm_response=response,
                    optional_params={
                        "model_name": model,
                        "prompt_version": version,
                        "topic_id": f"topic_{topic_idx+1}"
                    }
                )
    
    # We should have 3 topics × 3 models × 2 versions = 18 results
    assert len(lamoom_pipe.accumulated_results) == 18
    
    # Test multi-dimensional visualization: 
    # X-axis: topics, Lines: different models
    topic_model_path = os.path.join(temp_dir, "topic_by_model.png")
    lamoom_pipe.visualize_test_results(
        group_by="optional_param__topic_id",  # X-axis: topics
        group_line_by="optional_param__model_name",  # Lines: models
        show_statistics=True,
        save_path=topic_model_path
    )
    assert os.path.exists(topic_model_path)
    
    # Another visualization: 
    # X-axis: prompt versions, Lines: different models
    version_model_path = os.path.join(temp_dir, "version_by_model.png")
    lamoom_pipe.visualize_test_results(
        group_by="optional_param__prompt_version",  # X-axis: prompt versions
        group_line_by="optional_param__model_name",  # Lines: models
        show_statistics=True,
        save_path=version_model_path
    )
    assert os.path.exists(version_model_path)
    
    # And another way:
    # X-axis: models, Lines: prompt versions
    model_version_path = os.path.join(temp_dir, "model_by_version.png")
    lamoom_pipe.visualize_test_results(
        group_by="optional_param__model_name",  # X-axis: models
        group_line_by="optional_param__prompt_version",  # Lines: prompt versions
        save_path=model_version_path
    )
    assert os.path.exists(model_version_path)