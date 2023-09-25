import 'std/dotenv/load.ts'
import { retry } from 'std/async/retry.ts'
import { green, stripAnsiCode, yellow } from 'std/fmt/colors.ts'

const API_TOKEN = Deno.env.get('API_TOKEN')
const API_ROOT_URL = Deno.env.get('API_ROOT_URL')

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

function fmtPrimitive(x: string | number, type?: 'string' | 'number') {
	const t = type ?? typeof x
	return (t === 'string' ? green : yellow)((t === 'string' ? JSON.stringify : String)(x))
}

function format(r: Awaited<ReturnType<typeof query>>) {
	const entries = Object.entries(r.results)

	return `{
	${fmtPrimitive('source')}: ${fmtPrimitive(r.source)},
	${fmtPrimitive('results')}: [\n${
		entries.map(([k, v], i) =>
			`${i % 3 ? '' : '\n'}\t\t[${fmtPrimitive(v.toFixed(3), 'number')}, ${fmtPrimitive(k)}],`
		)
			.join('\n').slice(1)
	}\n\t],
}
`
}

const formatted = format(r)

console.info(formatted)
await Deno.writeTextFile('results.jsonc', stripAnsiCode(formatted))
console.info('Done')
