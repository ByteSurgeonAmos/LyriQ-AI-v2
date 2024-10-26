import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';

interface ChatRequest {
  message: string;
  document_id?: number;
  session_id?: string;
}

@Injectable({
  providedIn: 'root'
})
export class RapBotService {
  private apiUrl = 'http://localhost:8000/api';

  constructor(private http: HttpClient) { }

  // 1. Upload Document
  uploadDocument(file: File): Observable<any> {
    const formData = new FormData();
    formData.append('file', file);

    return this.http.post(`${this.apiUrl}/upload/`, formData);
  }

  // 2. Start New Chat
  startNewChat(message: string, documentId: number): Observable<any> {
    const headers = new HttpHeaders({ 'Content-Type': 'application/json' });
    const body: ChatRequest = {
      message,
      document_id: documentId
    };

    return this.http.post(`${this.apiUrl}/chat/`, body, { headers });
  }

  // 3. Continue Chat
  continueChat(sessionId: string, message: string): Observable<any> {
    const headers = new HttpHeaders({ 'Content-Type': 'application/json' });
    const body: ChatRequest = {
      session_id: sessionId,
      message
    };

    return this.http.post(`${this.apiUrl}/chat/`, body, { headers });
  }

  // 4. View Session History
  viewSessionHistory(sessionId: string): Observable<any> {
    return this.http.get(`${this.apiUrl}/chat/history/${sessionId}/`);
  }

  // 5. View All Sessions
  viewAllSessions(): Observable<any> {
    return this.http.get(`${this.apiUrl}/chat/history/`);
  }
}
