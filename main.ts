import 'std/dotenv/load.ts'
import { retry } from 'std/async/retry.ts'

const API_TOKEN = Deno.env.get('API_TOKEN')
const API_ROOT_URL = 'https://api-inference.huggingface.co/'

const url = new URL('models/sentence-transformers/stsb-xlm-r-multilingual', API_ROOT_URL)

type Payload = {
	inputs: {
		source_sentence: string
		sentences: string[]
	}
}

const { source, targets: _targets } = JSON.parse(await Deno.readTextFile('./input.json')) as {
	source: string
	targets: string[]
}

const targets = _targets
	.map((x) => x.trim())
	.filter(Boolean)

type Params = {
	source: string
	targets: string[]
}

async function query({ source, targets }: Params) {
	const headers = new Headers({
		'content-type': 'application/json',
		accept: 'application/json',
		authorization: `Bearer ${API_TOKEN}`,
	})

	const payload: Payload = {
		inputs: {
			source_sentence: source,
			sentences: targets.map((t) => t.replaceAll(/^\d+\. |[\[\]【】]/gu, '')),
		},
	}

	const results: number[] = await retry(async () => {
		const res = await fetch(url, {
			method: 'POST',
			headers,
			body: JSON.stringify(payload),
		})

		if (!res.ok) {
			throw new Error((await res.json()).error)
		}

		return await res.json()
	})

	return {
		source,
		results: Object.fromEntries(targets.map((t, i) => [t, results[i]])),
	}
}

const r = await query({ source, targets })

function format(r: Awaited<ReturnType<typeof query>>) {
	const entries = Object.entries(r.results)

	return `{
	"source": ${JSON.stringify(r.source)},
	"results": [\n${
		entries.map(([k, v]) => `\t\t[${v.toFixed(3)}, ${JSON.stringify(k)}],`)
			.join('\n')
	}\n\t],
}`
}

await Deno.writeTextFile('results.jsonc', format(r))

console.info(r)
console.info('Done')
