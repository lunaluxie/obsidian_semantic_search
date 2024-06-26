import {
    App,
    PluginSettingTab,
    Setting
} from 'obsidian';
import SemanticSearchPlugin from 'main';

export interface SemanticSearchSettings {
	embeddingModel: string;
	apiKey: string;
	chunkSize: number;
	resultCount: number;
    batchSize: number;
    lastUpdated: number;
}

export const DEFAULT_SETTINGS: SemanticSearchSettings = {
    embeddingModel: 'all-MiniLM-L6-v2',
    apiKey: '',
    chunkSize: 500,
    resultCount: 5,
    batchSize: 20,
    lastUpdated: 0,
};

export class SemanticSearchSettingTab extends PluginSettingTab {
    plugin: SemanticSearchPlugin;

    constructor(app: App, plugin: SemanticSearchPlugin) {
        super(app, plugin);
        this.plugin = plugin;
    }

    display(): void {
        const {containerEl} = this;
        containerEl.empty();

        new Setting(containerEl)
            .setName('Embedding Model')
            .setDesc('Select the model to use for embeddings.')
            .addDropdown(dropdown => dropdown
                .addOption('Xenova/all-MiniLM-L6-v2', 'all-MiniLM-L6-v2')
            // TODO:
            // .addOption('bert-base-uncased', 'bert-base-uncased')
            // .addOption('text-embedding-ada-002', 'text-embedding-ada-002')
            // .addOption('text-embedding-3-large', 'text-embedding-3-large')
            // .addOption('text-embedding-3-small', 'text-embedding-3-small')
                .setValue(this.plugin.settings.embeddingModel)
                .onChange(async (value) => {
                    //TODO: check if a vector store already exists. if it does, then warn against deleting it.
                    if (['text-embedding-ada-002', 'text-embedding-3-large', 'text-embedding-3-small'].includes(value)) {
                        apiKeySetting.setDisabled(false);
                    } else {
                        apiKeySetting.setDisabled(true);
                    }

                    this.plugin.settings.embeddingModel = value;
                    await this.plugin.saveSettings();
                })
            );

        const apiKeySetting = new Setting(containerEl)
            .setName('API Key')
            .setDesc('Enter your OpenAI API key here.')
            .addText(text => text
                .setPlaceholder('API Key')
                .setValue(this.plugin.settings.apiKey)
                .onChange(async (value) => {
                    this.plugin.settings.apiKey = value;
                    await this.plugin.saveSettings();
                })
                .then((cb) => {
                    cb.inputEl.style.width = '100%';
                }));

        new Setting(containerEl)
            .setName('Chunk Size')
            .setDesc('Enter the chunk size for the embeddings')
            .addText(text => text
                .setPlaceholder('Chunk Size')
                .setValue(this.plugin.settings.chunkSize.toString())
                .onChange(async (value) => {
                    const chunkSize = parseInt(value, 10);
                    if (Number.isInteger(chunkSize)) {
                        this.plugin.settings.chunkSize = chunkSize;
                    } else {
                        this.plugin.settings.chunkSize = DEFAULT_SETTINGS.chunkSize;
                    }
                    await this.plugin.saveSettings();
                })
            );

        new Setting(containerEl)
            .setName('Search Result Count')
            .setDesc('Enter the number of similar documents to return')
            .addText(text => text
                .setPlaceholder('Search Result Count')
                .setValue(this.plugin.settings.resultCount.toString())
                .onChange(async (value) => {
                    const resultCount = parseInt(value, 10);
                    if (Number.isInteger(resultCount)) {
                        this.plugin.settings.resultCount = resultCount;
                    } else {
                        this.plugin.settings.resultCount = DEFAULT_SETTINGS.resultCount;
                    }
                    await this.plugin.saveSettings();
                })
            );

        new Setting(containerEl)
            .setName('Embedding Batch Size')
            .setDesc('Enter the batch size for embeddings')
            .addText(text => text
                .setPlaceholder('Embedding Batch Size')
                .setValue('500')
                .setDisabled(true)
                .onChange(async (value) => {
                    const batchSize = parseInt(value, 20);
                    if (Number.isInteger(batchSize)) {
                        this.plugin.settings.batchSize = batchSize;
                    } else {
                        this.plugin.settings.batchSize = DEFAULT_SETTINGS.batchSize;
                    }
                    await this.plugin.saveSettings();
                })
            );

        new Setting(containerEl)
            .setName('Last Updated')
            .setDesc('Last time the embeddings were updated: ' + new Date(this.plugin.settings.lastUpdated).toLocaleString());
    }
}