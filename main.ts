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

const source = 'That is a happy person'
const targets = `

1. That is a happy person
1. Esa es una persona feliz
1. 那是一位开心的人

2. [that] is a happy person
2. [esa] es una persona feliz
2. 那是一位【開心】的人

3. [This] is a happy person
3. [Esta] es una persona feliz
3. 【这】是一位开心的人

4. That is a [very] happy person
4. Esa es una persona [muy] feliz
4. 那是一位【非常】开心的人

5. That is an [unhappy] person
5. Esa es una persona [infeliz]
5. 那是一位【不开心】的人

6. That is a [very unhappy] person
6. Esa es una persona [muy infeliz]
6. 那是一位【非常不开心】的人

7. That is [not] a happy person
7. Esa [no] es una persona feliz
7. 那【不】是一位开心的人

8. That is a happy [dog]
8. Ese es un [perro] feliz
8. 那是一条开心的【狗】

`
	.split('\n')
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
			sentences: targets.map((t) => t.replaceAll(/^\d+\. |[\[\]【】]/gmu, '')),
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
