# Copyright (c) 2021, EleutherAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
check implementation of NeoXArgs for duplication errors (would overwrite)
"""
import pytest


@pytest.mark.cpu
def test_neoxargs_duplicates():
    """
    tests that there are no duplicates among parent classes of NeoXArgs
    """
    from savanna import NeoXArgs

    assert NeoXArgs.validate_keys(), "test_neoxargs_duplicates"
