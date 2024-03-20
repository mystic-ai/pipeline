curl -X POST 'http://localhost:14300/v4/runs' \
--header 'Content-Type: application/json' \
--data '{
	"inputs":
		[
			{
				"type": "array",
				"value": ["hey there"]
			},
			{
				"type": "dictionary",
				"value": {
				}
			}
		]
	}
' -N
