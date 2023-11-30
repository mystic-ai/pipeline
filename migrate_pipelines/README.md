Copy old role ids by manually changing them in the new db: we only have 3 roles so this should be quick.

---

Export the following tables from old db:

1) user
2) user role
3) user billing
4) token

 Make sure to export with headers,

---

Use the `GET_DOUBLE_USERNAMES` query to get double users, export to `old_csvs/doubles.csv` and remove any that you want to keep (you can only choose one for each duplicate!)
i.e. Users in `doubles.csv` will be removed from all the other tables.

Use the `GET_PIPELINE_FAMILIES_UNIQUE` query to get pipeline families filtered to lower case, export to `old_csvs/pipeline_family.csv`.

---

Run the `prepare_csvs.py` script to convert usernames and remove the double users.

---

Import the new csvs in the same order as you exported.

---

Export the pipeline meta table to `old_csvs/pipeline_meta.csv`.

Export the pointer table to `old_csvs/pointer.csv`.

Export the environment table to `old_csvs/environment.csv`

Run the pipeline data query in `script.py` and export the results to `old_csvs/pipelines.csv`.

---

Login to the target pcore stack.

Change directory to `./migrate_pipelines` if not already.

Set env variables `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` from the v3 catalyst secrets. Leave them in their base64 encoded form.

Run `script.py`.

---

Truncate the pointer table.

Import `new_csvs/pipeline_metas.csv` and `ne_csvs/pointer.csv` into the new db.
