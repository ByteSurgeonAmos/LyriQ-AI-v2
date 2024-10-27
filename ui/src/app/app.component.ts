import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { RapBotService } from './rap-bot.service';

interface Message {
  sender: 'user' | 'bot';
  content: string;
  sentiment?: 'positive' | 'negative' | 'neutral';
  error?: boolean;
}

interface UploadedDoc {
  name: string;
  type: string;
  id?: number;
}

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
  messages: Message[] = [];
  uploadedDocs: UploadedDoc[] = [];
  currentSessionId: string | null = null;
  
  // New state management properties
  isLoading = {
    upload: false,
    message: false
  };
  errors = {
    upload: null as string | null,
    message: null as string | null
  };
  uploadProgress = 0;

  constructor(private rapBotService: RapBotService) {}

  setMode(mode: 'sentiment' | 'chat') {
    this.activeMode = mode;
    this.clearErrors();
  }

  getInputPlaceholder(): string {
    return this.activeMode === 'sentiment'
      ? (this.selectedLanguage === 'en' ? 'Enter lyrics to analyze...' : 'Geben Sie Songtexte zur Analyse ein...')
      : (this.selectedLanguage === 'en' ? 'Ask questions about your documents...' : 'Stellen Sie Fragen zu Ihren Dokumenten...');
  }

  // Clear all error states
  clearErrors() {
    this.errors.upload = null;
    this.errors.message = null;
  }

  // Handle file selection via input
  onFileSelect(event: Event) {
    const input = event.target as HTMLInputElement;
    if (input.files) {
      this.handleFiles(input.files);
    }
  }

  // Trigger file input click
  triggerFileInput() {
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = '.txt,.pdf';
    fileInput.multiple = true;
    fileInput.onchange = (e) => this.onFileSelect(e);
    fileInput.click();
  }

  handleDrop(event: DragEvent) {
    event.preventDefault();
    event.stopPropagation();
    const files = event.dataTransfer?.files;
    if (files) {
      this.handleFiles(files);
    }
  }

  handleDragOver(event: DragEvent) {
    event.preventDefault();
    event.stopPropagation();
  }

  handleFiles(files: FileList) {
    this.isLoading.upload = true;
    this.clearErrors();
    this.uploadProgress = 0;

    const validFiles = Array.from(files).filter(file => 
      file.type === 'text/plain' || file.type === 'application/pdf'
    );

    if (validFiles.length === 0) {
      this.errors.upload = 'Please upload only TXT or PDF files.';
      this.isLoading.upload = false;
      return;
    }

    const totalFiles = validFiles.length;
    let completedFiles = 0;

    validFiles.forEach(file => {
      this.rapBotService.uploadDocument(file).subscribe({
        next: response => {
          const documentId = response.document_id;
          this.uploadedDocs.push({
            name: file.name,
            type: file.type,
            id: documentId
          });
          completedFiles++;
          this.uploadProgress = (completedFiles / totalFiles) * 100;
          
          if (completedFiles === totalFiles) {
            this.isLoading.upload = false;
            this.uploadProgress = 0;
          }
        },
        error: error => {
          this.errors.upload = `Error uploading ${file.name}: ${error.message}`;
          this.isLoading.upload = false;
        }
      });
    });
  }

  async sendMessage() {
    if (!this.messageInput.trim()) return;
    
    this.isLoading.message = true;
    this.clearErrors();
    
    const userMessage = this.messageInput;
    this.messages.push({ sender: 'user', content: userMessage });
    this.messageInput = ''; // Clear input immediately

    if (this.activeMode === 'sentiment') {
      const doc = this.uploadedDocs[0];
      
      if (!doc?.id) {
        this.errors.message = 'Please upload a document first.';
        this.isLoading.message = false;
        return;
      }

      this.rapBotService.startNewChat(userMessage, doc.id).subscribe({
        next: response => {
          this.currentSessionId = response.session_id;
          this.messages.push({
            sender: 'bot',
            content: '',
            sentiment: response.response
          });
          this.isLoading.message = false;
        },
        error: error => {
          this.messages.push({
            sender: 'bot',
            content: 'Sorry, I encountered an error analyzing the sentiment.',
            error: true
          });
          this.errors.message = `Error: ${error.message}`;
          this.isLoading.message = false;
        }
      });
    } else if (this.currentSessionId) {
      this.rapBotService.continueChat(this.currentSessionId, userMessage).subscribe({
        next: response => {
          this.messages.push({
            sender: 'bot',
            content: response.response
          });
          this.isLoading.message = false;
        },
        error: error => {
          this.messages.push({
            sender: 'bot',
            content: 'Sorry, I encountered an error processing your message.',
            error: true
          });
          this.errors.message = `Error: ${error.message}`;
          this.isLoading.message = false;
        }
      });
    } else {
      this.errors.message = 'No active session. Please start with sentiment analysis first.';
      this.isLoading.message = false;
    }
  }

  removeDocument(docIndex: number) {
    this.uploadedDocs.splice(docIndex, 1);
    if (this.uploadedDocs.length === 0) {
      this.currentSessionId = null;
    }
  }
}