import datetime

from pipeline.cloud.pipelines import map_pipeline_mp

if __name__ == "__main__":
    start_time = datetime.datetime.now()
    input_array = list(range(100))
    result = map_pipeline_mp(
        input_array,
        "9",
        pool_size=100,
    )

    end_time = datetime.datetime.now()

    total_time = (end_time - start_time).total_seconds()

    print(f"Total of {len(input_array)} parallel tasks")
    print(
        f"Total time taken: {total_time} "
        f"({int(total_time * 1e4/len(input_array))/1e4} /req)"
        ", result: {result}"
    )
