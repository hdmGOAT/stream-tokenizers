import { StreamingTokenizer, WordLevel } from '../../'

describe('StreamingTokenizer', () => {
  it('can be imported from the package root', () => {
    expect(StreamingTokenizer).toBeDefined()
  })

  it('processes chunks and drains tokens for WordLevel model', () => {
    const model = WordLevel.init(
      {
        Hello: 0,
        world: 1,
        '[UNK]': 2,
      },
      { unkToken: '[UNK]' },
    )

    const stream = new StreamingTokenizer(model, 1024)

    const config = stream.config()
    expect(config).toBeDefined()
    expect(typeof config.requiresWordBoundaries).toBe('boolean')
    expect(typeof config.lookaheadBytes).toBe('number')
    expect(typeof config.canEmitIncrementally).toBe('boolean')
    expect(typeof config.minChunkSize).toBe('number')

    stream.processChunk(Buffer.from('Hello '))
    stream.processChunk(Buffer.from('world'))
    stream.finalize()

    const tokens = stream.drainTokens()
    expect(tokens).toHaveLength(2)
    expect(tokens.map((token) => token.value)).toEqual(['Hello', 'world'])

    for (const token of tokens) {
      expect(typeof token.id).toBe('number')
      expect(Array.isArray(token.offsets)).toBe(true)
      expect(token.offsets).toHaveLength(2)
    }

    expect(stream.drainTokens()).toEqual([])
  })
})
