import csv

GET_DOUBLE_USERNAMES = """with doubles as (SELECT lower(username) as u, count(*) as c from catalyst.user group by u having count(*) > 1),
doubles_users as (SELECT username, id, created_at as c_a from catalyst.user join doubles on lower(username) = u),
run_counts as (SELECT username, id, c_a, count(run_id) as run_count from doubles_users left join catalyst.run_meta on id = run_meta.user_id group by (username, id, c_a)),
pipeline_counts as (SELECT username, id, count(pipeline_id) as pipeline_count from doubles_users left join catalyst.pipeline_meta on id = pipeline_meta.user_id group by (username, id, c_a)),
latest_run as (SELECT username, id, max(run_meta.created_at) as latest_run from doubles_users left join catalyst.run_meta on id = run_meta.user_id group by (username, id, c_a))
select run_counts.username, run_counts.id, c_a as created_at, pipeline_count, run_count, latest_run from run_counts join pipeline_counts on run_counts.username = pipeline_counts.username join latest_run on run_counts.username = latest_run.username order by run_counts.username"""

GET_PIPELINE_FAMILIES_UNIQUE = """select distinct on (lower(name)) name, '0' as run_count, created_at, updated_at from catalyst.pipeline_family"""


def parse_csv(f):
    rows = []
    with open(f) as csvfile:
        reader = csv.reader(csvfile)
        headers = reader.__next__()  # skip header
        for row in reader:
            rows.append(row)
    return rows, headers


def write_csv(f, rows, header):
    with open(f, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def filter_users(rows: list, id_i, ban_list, ban_list_i):
    ban_list_ids = [b[ban_list_i] for b in ban_list]
    return list(filter(lambda x: x[id_i] not in ban_list_ids, rows))


if __name__ == "__main__":
    doubles, double_headers = parse_csv("./old_csvs/doubles.csv")
    users, user_headers = parse_csv("./old_csvs/user.csv")
    user_roles, user_role_headers = parse_csv("./old_csvs/user_role.csv")
    user_billings, user_billing_headers = parse_csv("./old_csvs/user_billing.csv")
    tokens, token_headers = parse_csv("./old_csvs/token.csv")
    pipeline_families, pipeline_family_headers = parse_csv(
        "./old_csvs/pipeline_family.csv"
    )

    new_users = filter_users(users, 5, doubles, 1)
    new_users = map(lambda x: [x[0].lower()] + x[1:], new_users)
    new_user_roles = filter_users(user_roles, 0, doubles, 1)
    new_user_billings = filter_users(user_billings, 3, doubles, 1)
    new_tokens = filter_users(tokens, 5, doubles, 1)
    new_pipeline_families = map(lambda x: [x[0].lower()] + x[1:], pipeline_families)

    write_csv("./new_csvs/user.csv", new_users, user_headers)
    write_csv("./new_csvs/user_role.csv", new_user_roles, user_role_headers)
    write_csv("./new_csvs/user_billing.csv", new_user_billings, user_billing_headers)
    write_csv("./new_csvs/token.csv", new_tokens, token_headers)
    write_csv(
        "./new_csvs/pipeline_family.csv", new_pipeline_families, pipeline_family_headers
    )
