import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { RapBotService } from './rap-bot.service'; // Ensure service is imported here

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, FormsModule, CommonModule],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  activeMode: 'sentiment' | 'chat' = 'sentiment';
  messageInput = '';
  selectedLanguage: 'en' | 'de' = 'en';
  messages: Array<{
    sender: 'user' | 'bot';
    content: string;
    sentiment?: 'positive' | 'negative' | 'neutral';
  }> = [];
  uploadedDocs: Array<{
    name: string;
    type: string;
    id?: number;
  }> = [];
  currentSessionId: string | null = null;

  constructor(private rapBotService: RapBotService) { }

  setMode(mode: 'sentiment' | 'chat') {
    this.activeMode = mode;
  }

  getInputPlaceholder(): string {
    return this.activeMode === 'sentiment'
      ? (this.selectedLanguage === 'en' ? 'Enter lyrics to analyze...' : 'Geben Sie Songtexte zur Analyse ein...')
      : (this.selectedLanguage === 'en' ? 'Ask questions about your documents...' : 'Stellen Sie Fragen zu Ihren Dokumenten...');
  }

  handleDrop(event: DragEvent) {
    event.preventDefault();
    const files = event.dataTransfer?.files;
    if (files) {
      this.handleFiles(files);
    }
  }

  handleFiles(files: FileList) {
    Array.from(files).forEach(file => {
      this.rapBotService.uploadDocument(file).subscribe({
        next: response => {
          const documentId = response.document_id; 
          this.uploadedDocs.push({
            name: file.name,
            type: file.type,
            id: documentId
          });
        },
        error: error => console.error('Error uploading file:', error)
      });
    });
  }

  sendMessage() {
    if (!this.messageInput.trim()) return;

    this.messages.push({ sender: 'user', content: this.messageInput });

    console.log(this.activeMode);
    
    if (this.activeMode === 'sentiment') {
      // For sentiment analysis, start a new chat with the document ID
      const doc = this.uploadedDocs[0]; // Assuming the first uploaded doc for simplicity
      console.log(doc);
      
      if (doc && doc.id) {
        this.rapBotService.startNewChat(this.messageInput, doc.id).subscribe({
          next: response => {
            this.currentSessionId = response.session_id; // Save session ID for future messages
            this.messages.push({
              sender: 'bot',
              content: 'Based on the analysis of these lyrics...',
              sentiment: response.response 
            });
          },
          error: error => console.error('Error starting chat:', error)
        });
      }
    } else if (this.currentSessionId) {
      // Continue chat if a session ID exists
      this.rapBotService.continueChat(this.currentSessionId, this.messageInput).subscribe({
        next: response => {
          this.messages.push({
            sender: 'bot',
            content: response.response
          });
        },
        error: error => console.error('Error continuing chat:', error)
      });
    } else {
      console.error('No session ID found. Start a new chat first.');
    }

    this.messageInput = '';
  }
}
