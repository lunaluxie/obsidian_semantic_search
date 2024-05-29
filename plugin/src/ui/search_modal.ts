import { App, SuggestModal } from 'obsidian';
import { SemanticSearchSettings } from 'src/ui/settings';
import { queryNoteChunks } from 'src/api/semantic_search_service';
import { Chunk } from 'src/interfaces';


export class SearchModal extends SuggestModal<Chunk> {
    settings: SemanticSearchSettings;
    vaultPath: string;

    constructor(app: App, settings: SemanticSearchSettings, vaultPath: string) {
        super(app);
        this.settings = settings;
        this.vaultPath = vaultPath;
    }

    async getSuggestions(query: string): Promise<Chunk[]> {
        const queryDetails = {
            model: this.settings.embeddingModel,
            vaultPath: this.vaultPath,
            pluginPath: this.vaultPath + '/.obsidian/plugins/semantic_search',
            query: query,
            searchResultsCount: this.settings.resultCount
        };

        // filter out duplicate file_paths from the suggestions
        // since we cannot open deep links anyway.
        let suggestions = (await queryNoteChunks(queryDetails)).filter((value, index, self) => {
            return self.findIndex(v => v.file_path === value.file_path) === index;
        })

        return suggestions;
    }

    renderSuggestion({ file_name, file_path, text_chunk }: Chunk, el: HTMLElement) {
        el.createEl('span', { text: file_name, cls: 'omnisearch-result__title' });
        el.createEl('span', { text: file_path, cls: 'omnisearch-result__folder-path' });
        el.createEl('div', { text: text_chunk, cls: 'omnisearch-result__body' });
    }

    onChooseSuggestion({ file_path }: Chunk, evt: MouseEvent | KeyboardEvent) {
        console.log(evt);
        this.app.workspace.openLinkText(file_path, '');
    }
}