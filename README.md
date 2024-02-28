# Obsidian Semantic Search

Simple interface for semantic search over markdown files in an Obsidian vault. The project is an experimental and work in progress. A longer description of the implementation can be found [here]().

## Usage

Currently there's seven commands in the plugin interface: Info, Unindexed Files, Seach, Embed File, Reset Index, Embed Vault, Update Index:
- **Info** displays the number of files embedded.
- **Unindexed Files** returns the names of the files that have not been embedded.
- **Search** semantically search for terms and expression in the vualt.
- **Embed File** add embedding for the currently open file to the vector datastore.
- **Reset Index** removes all the embeddings from the vector datastore.
- **Embed Vault** adds all the markdown files in the vault to the vector datastore.
- **Update Index** updates embedding index for vector datastore.

### Embedding a Single File

![Embed File](./docs/embed_file.gif)

### Embedding the Entire Vault

![Embed Vault](./docs/embed_vault.gif)

### Searching

![Search](./docs/search.gif)

## Install & Setup

The current design requires a server to be running to handle the search requests. The server is a simple node app that implements an express server. You can configure the server to run on a specific port.

To run the server in the server in the background, first clone the repository and install the dependencies and then run the prod script:

```bash
$ cd obsidian-semantic-search
$ npm install
$ npm run prod
```
