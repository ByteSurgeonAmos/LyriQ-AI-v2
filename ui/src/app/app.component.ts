import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, FormsModule, CommonModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
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
  }> = [];

  setMode(mode: 'sentiment' | 'chat') {
    this.activeMode = mode;
  }
  getInputPlaceholder(): string {
    if (this.activeMode === 'sentiment') {
      return this.selectedLanguage === 'en' ? 'Enter lyrics to analyze...' : 'Geben Sie Songtexte zur Analyse ein...';
    } else {
      return this.selectedLanguage === 'en' ? 'Ask questions about your documents...' : 'Stellen Sie Fragen zu Ihren Dokumenten...';
    }
  }
  
  handleDrop(event: DragEvent) {
    event.preventDefault();
    const files = event.dataTransfer?.files;
    if (files) {
      this.handleFiles(files);
    }
  }

  handleFiles(files: FileList) {
    // Implement file upload logic
    Array.from(files).forEach(file => {
      this.uploadedDocs.push({
        name: file.name,
        type: file.type
      });
    });
  }

  sendMessage() {
    if (!this.messageInput.trim()) return;

    this.messages.push({
      sender: 'user',
      content: this.messageInput
    });

    // Simulate response
    setTimeout(() => {
      if (this.activeMode === 'sentiment') {
        this.messages.push({
          sender: 'bot',
          content: 'Based on the analysis of these lyrics...',
          sentiment: 'positive' // This would come from the actual model
        });
      } else {
        this.messages.push({
          sender: 'bot',
          content: 'Let me search through your documents...'
        });
      }
    }, 1000);

    this.messageInput = '';
  }
}