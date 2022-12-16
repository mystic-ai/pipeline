import dill

from pipeline import Pipeline, PipelineCloud, PipelineFile, pipeline_function

pcloud = PipelineCloud()


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

res = pcloud.run_pipeline(rpl_remote, [])
print(f"Remote result:{res.result_preview}")


@pipeline_function
def return_file_contents(mf: PipelineFile) -> str:
    print(mf.path)
    with open(mf.path, "rb") as tmp:
        return dill.load(tmp)


with Pipeline("pf-test") as builder:
    pfile = PipelineFile(remote_id=res.result.id)
    builder.add_variables(pfile)
    result = return_file_contents(pfile)
    builder.output(result)

pf_test = Pipeline.get_pipeline("pf-test")
pcloud.download_remotes(pf_test)

final_result = pf_test.run()

print(f"Local result: {final_result[0]}")
