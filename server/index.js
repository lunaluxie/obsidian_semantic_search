import express from 'express'
import cors from 'cors'
import { VectorStore } from './src/db/vector_store.js'
import { buildDbPath } from './src/util.js'

const app = express()
app.use(express.json())
app.use(cors())
const PORT = process.env.PORT || 3003

app.get('/check_status', (req, res) => { res.sendStatus(200) })

app.post('/info', (req, res) => {
  const vectDb = new VectorStore(buildDbPath(req.body))

  vectDb.info((err, result) => {
    if (err) {
      res.sendStatus(500)
    } else {
      res.status(200).json(result)
    }
  })
})

app.post('/configure', (req, res) => {
  try {
    new VectorStore(buildDbPath(req.body)).configure(req.body.model)
    res.sendStatus(200)
  } catch (error) {
    console.error(error)
    res.sendStatus(500)
  }
})

app.post('/embed_file', async (req, res) => {
  const vectDb = new VectorStore(buildDbPath(req.body))

  try {
    vectDb.embedFile(
      req.body.chunkSize,
      50,
      req.body.model,
      req.body.fileName,
      req.body.filePath,
      req.body.vaultPath
    )

    res.sendStatus(200)
  } catch (error) {
    console.error(error)
    res.sendStatus(500)
  }
})

app.post('/embed_batch', async (req, res) => {
  const vectDb = new VectorStore(buildDbPath(req.body))
  try {
    await vectDb.embedBatch(
      req.body.chunkSize,
      50,
      req.body.model,
      req.body.files,
      req.body.vaultPath
    )

    res.sendStatus(200)
  } catch (error) {
    console.error(error)
    res.sendStatus(500)
  }
})

app.post('/delete_files_embedding', async (req, res) => {
  const vectDb = new VectorStore(buildDbPath(req.body))
  try {
    console.log("deleting files embeddings for ")
    console.log(req.body.filePaths)
    await vectDb.deleteFilesEmbedding(
      req.body.filePaths
    )

    res.sendStatus(200)
  } catch (error) {
    console.error(error)
    res.sendStatus(500)
  }
})

app.post('/embedded_files', async (req, res) => {
  try {
    const fileNames = new VectorStore(buildDbPath(req.body)).fileNames()
    console.log('fileNames: ', fileNames)
    res.status(200).json(fileNames)
  } catch (error) {
    console.error(error)
    res.sendStatus(500)
  }
})

app.post('/reset', async (req, res) => {
  try {
    new VectorStore(buildDbPath(req.body)).reset()
    res.sendStatus(200)
  } catch (error) {
    console.error(error)
    res.sendStatus(500)
  }
})

app.post('/update_index', (req, res) => {
  console.log(req.body)
  try {
    new VectorStore(buildDbPath(req.body)).updateIndex()
    res.sendStatus(200)
  } catch (error) {
    console.error(error)
    res.sendStatus(500)
  }
})

app.post('/query', async (req, res) => {
  if (!req.body.query || req.body.query.trim() === '') {
    res.status(200).json([])
    return
  }

  const vectDb = new VectorStore(buildDbPath(req.body))
  try {
    const searchResults = await vectDb.query(
      req.body.query,
      req.body.searchResultsCount,
      req.body.model
    )

    res.status(200).json(searchResults)
  } catch (error) {
    console.error(error)
    res.sendStatus(500)
  }
})

app.listen(PORT, () => {
  console.log(`Semantic Search server running on port ${PORT}`)
})
