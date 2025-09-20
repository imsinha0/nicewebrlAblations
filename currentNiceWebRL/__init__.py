from currentNiceWebRL.dataframe import DataFrame
from currentNiceWebRL.dataframe import concat_dataframes

from currentNiceWebRL.utils import toggle_fullscreen
from currentNiceWebRL.utils import check_fullscreen
from currentNiceWebRL.utils import clear_element
from currentNiceWebRL.utils import wait_for_button_or_keypress
from currentNiceWebRL.utils import retry_with_exponential_backoff
from currentNiceWebRL.utils import basic_javascript_file
from currentNiceWebRL.utils import multihuman_javascript_file
from currentNiceWebRL.utils import initialize_user
from currentNiceWebRL.utils import get_user_session_minutes
from currentNiceWebRL.utils import broadcast_message
from currentNiceWebRL.utils import read_msgpack_records
from currentNiceWebRL.utils import write_msgpack_record
from currentNiceWebRL.utils import read_all_records
from currentNiceWebRL.utils import read_all_records_sync
from currentNiceWebRL.utils import get_user_lock
from currentNiceWebRL.utils import prevent_default_spacebar_behavior
from currentNiceWebRL.utils import user_data_file
from currentNiceWebRL.utils import user_metadata_file
from currentNiceWebRL.utils import save_metadata
from currentNiceWebRL.utils import get_progress
from currentNiceWebRL.utils import load_data

import os

# Check environment variable to determine which nicejax module to use
ABLATION_MODE = os.getenv("ABLATION_MODE", "normal")

if ABLATION_MODE == "ablation1" or ABLATION_MODE == "ablation4":
    # Import from ablation1nicejax.py
    from ablation1.ablation1nicejax import get_rng
    from ablation1.ablation1nicejax import new_rng
    from ablation1.ablation1nicejax import match_types
    from ablation1.ablation1nicejax import make_serializable
    from ablation1.ablation1nicejax import base64_npimage
    from ablation1.ablation1nicejax import StepType
    from ablation1.ablation1nicejax import TimeStep
    from ablation1.ablation1nicejax import EnvParams
    from ablation1.ablation1nicejax import TimestepWrapper
    from ablation1.ablation1nicejax import JaxWebEnv
    from ablation1.ablation1nicejax import get_size
elif ABLATION_MODE == "ablation3":
    # Import from ablation3nicejax.py
    from ablation3.ablation3nicejax import get_rng
    from ablation3.ablation3nicejax import new_rng
    from ablation3.ablation3nicejax import match_types
    from ablation3.ablation3nicejax import make_serializable
    from ablation3.ablation3nicejax import base64_npimage
    from ablation3.ablation3nicejax import StepType
    from ablation3.ablation3nicejax import TimeStep
    from ablation3.ablation3nicejax import EnvParams
    from ablation3.ablation3nicejax import TimestepWrapper
    from ablation3.ablation3nicejax import JaxWebEnv
    from ablation3.ablation3nicejax import get_size
else:
    # Import from normal nicejax.py
    from currentNiceWebRL.nicejax import get_rng
    from currentNiceWebRL.nicejax import new_rng
    from currentNiceWebRL.nicejax import match_types
    from currentNiceWebRL.nicejax import make_serializable
    # from nicewebrl.nicejax import deserialize
    from currentNiceWebRL.nicejax import base64_npimage
    from currentNiceWebRL.nicejax import StepType
    from currentNiceWebRL.nicejax import TimeStep
    from currentNiceWebRL.nicejax import EnvParams
    from currentNiceWebRL.nicejax import TimestepWrapper
    from currentNiceWebRL.nicejax import JaxWebEnv
    from currentNiceWebRL.nicejax import get_size


from currentNiceWebRL.stages import EnvStageState
from currentNiceWebRL.stages import StageStateModel
from currentNiceWebRL.stages import Stage
from currentNiceWebRL.stages import FeedbackStage
from currentNiceWebRL.stages import EnvStage
from currentNiceWebRL.stages import Block
from currentNiceWebRL.stages import prepare_blocks
from currentNiceWebRL.stages import generate_stage_order
from currentNiceWebRL.stages import time_diff
from currentNiceWebRL.stages import broadcast_metadata

from currentNiceWebRL.experiment import Experiment
from currentNiceWebRL.experiment import SimpleExperiment
from currentNiceWebRL.container import Container

from currentNiceWebRL.logging import get_logger
from currentNiceWebRL.logging import setup_logging

from currentNiceWebRL.data_analysis import compute_reaction_time
from currentNiceWebRL.data_analysis import time_diff