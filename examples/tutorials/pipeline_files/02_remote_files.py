##########
# PipelineFiles Tutorial 02 - Remote files
#
# This tutorial demonstrates how to execute a run remotely and use the result
# as part of a subsequent run both locally and remotely.
#
##########

import dill

from pipeline import Pipeline, PipelineCloud, PipelineFile, pipeline_function

pcloud = PipelineCloud()

"""

Initially we will create a pipeline that generates a random number and returns it.
We will access that random number later on via the use of a PipelineFile object both
locally and remotely.

"""


@pipeline_function
def output_random() -> float:
    import numpy as np

    return float(np.random.rand())


with Pipeline("rand-num") as builder:
    res = output_random()
    builder.output(res)

rpl = Pipeline.get_pipeline("rand-num")
rpl_remote = pcloud.upload_pipeline(rpl)
print(rpl_remote.id)

# The 'initial_run' output is the random number that we will access later on
initial_run = pcloud.run_pipeline(rpl_remote, [])

"""

Now that we have generated a random number as the output of a run we will use its file
as a PipelineFile in another pipeline and demonstrate the retrieval of it's contents.

"""


@pipeline_function
def return_file_contents(mf: PipelineFile) -> str:
    with open(mf.path, "rb") as tmp:
        return dill.load(tmp)[0]


with Pipeline("pf-test") as builder:
    pfile = PipelineFile(remote_id=initial_run.result.id)
    builder.add_variables(pfile)
    result = return_file_contents(pfile)
    builder.output(result)


pf_test = Pipeline.get_pipeline("pf-test")
pf_remote = pcloud.upload_pipeline(pf_test)
pcloud.download_remotes(pf_test)  # To run the locally you must download the file data

# Perform local and remote runs
local_result = pf_test.run()
remote_result = pcloud.run_pipeline(pf_remote, [])

# All outputs are the same meaning that we have succesfully passed the initial result on
print("---------- Final results ----------")
print(f"Initial run result: {initial_run.result_preview}")
print(f"Local output: {local_result}")
print(f"Remote output: {remote_result.result_preview}")
