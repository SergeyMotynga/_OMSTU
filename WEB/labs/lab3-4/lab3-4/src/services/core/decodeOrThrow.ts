import * as t from 'io-ts'
import { PathReporter } from 'io-ts/PathReporter'
import { pipe } from 'fp-ts/function'
import * as E from 'fp-ts/Either'

export function decodeOrThrow<C extends t.Mixed>(codec: C, data: unknown): t.TypeOf<C> {
  const result = codec.decode(data)
  return pipe(
    result,
    E.fold(
      errors => {
        throw new Error(PathReporter.report(E.left(errors)).join('\n'))
      },
      value => value
    )
  )
}