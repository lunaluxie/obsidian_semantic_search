export const createEmbeddingsTable = `
  CREATE TABLE IF NOT EXISTS note_chunks (
    file_name VARCHAR(255),
    file_path TEXT,
    text_chunk TEXT,
    embedding BLOB
  )
`

export const createVirtualTable = `
  CREATE VIRTUAL TABLE IF NOT EXISTS vss_note_chunks USING vss0(
    embedding(384)
  )
`

export const insertEmbeddingsIntoVSS = `
  INSERT INTO vss_note_chunks(rowid, embedding)
  SELECT rowid, embedding
  FROM note_chunks
`

export const insertFileEmbeddingsIntoVSS = `
  INSERT INTO vss_note_chunks(rowid, embedding)
  SELECT rowid, embedding
  FROM note_chunks
  WHERE file_path = ?
`

export const insertNoteChunk = `
  INSERT INTO note_chunks (file_name, file_path, text_chunk, embedding)
  VALUES (?, ?, ?, ?)
`

export function insertMultipleNoteChunks (chunkCount) {
  const placeholders = Array(chunkCount).fill('(?, ?, ?, ?)').join(',')
  return `INSERT INTO note_chunks (file_name, file_path, text_chunk, embedding)
    VALUES ${placeholders}`
}

export const deleteFileFromVss = `
  DELETE FROM vss_note_chunks
  WHERE rowid IN (
    SELECT rowid FROM note_chunks WHERE file_path = ?
  )
`

export function deleteFilesFromVss(filePaths) {
  const placeholders = filePaths.map(() => '?').join(',')
  return `DELETE FROM vss_note_chunks
    WHERE rowid IN (
      SELECT rowid FROM note_chunks WHERE file_path IN (${placeholders})
    )`
}

export const deleteFromVss = `
  DELETE FROM vss_note_chunks
`

export const deleteFileFromNoteChunks = 'DELETE FROM note_chunks WHERE file_path = ?'

export function deleteFilesFromNoteChunks (filePaths) {
  const placeholders = filePaths.map(() => '?').join(',')
  return `DELETE FROM note_chunks
    WHERE file_path in (${placeholders})`
}

export const embeddingsQuery = `
  WITH matches AS (
    SELECT
      rowid,
      distance
    FROM vss_note_chunks
    WHERE vss_search(
      embedding,
      ?
    )
    LIMIT ?
  )
  SELECT
    note_chunks.rowid,
    note_chunks.file_name,
    note_chunks.file_path,
    note_chunks.text_chunk,
    matches.distance
  FROM matches
  INNER JOIN note_chunks ON note_chunks.rowid = matches.rowid;`

export const countVss = 'SELECT count(1) FROM vss_note_chunks'
export const countChunks = 'SELECT count(1) FROM note_chunks'
