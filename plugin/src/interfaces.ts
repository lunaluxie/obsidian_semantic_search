export interface Chunk {
  file_name: string;
  file_path: string;
  text_chunk: string;
}

export interface DbDetails {
  vaultPath: string;
  pluginPath: string;
}

export interface FileDetails {
  fileName: string;
  filePath: string;
}

export interface QueryDetails {
  model: string;
  vaultPath: string;
  pluginPath: string;
  query: string;
  searchResultsCount: number;
}