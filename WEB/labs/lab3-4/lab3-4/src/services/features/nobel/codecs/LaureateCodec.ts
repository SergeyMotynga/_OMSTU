import * as t from 'io-ts'

export const NobelPrizeReferenceCodec = t.type({
  awardYear: t.string,
})

export const LaureateCodec = t.type({
  knownName: t.type({ en: t.string }),
  fullName: t.union([ t.undefined, t.type({ en: t.string })]),
  birth: t.union([ t.undefined, t.type({ date: t.string })]),
  nobelPrizes: t.union([ t.undefined, t.array(NobelPrizeReferenceCodec)]) })

export type Laureate = t.TypeOf<typeof LaureateCodec>