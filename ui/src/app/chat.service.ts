// chat.service.ts
import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, of } from 'rxjs';
import { map, catchError } from 'rxjs/operators';

export type SentimentType = 'positive' | 'negative' | 'neutral';
export type LanguageType = 'en' | 'de';

export interface ChatMessage {
  sender: 'user' | 'bot';
  content: string;
  sentiment?: SentimentType;
  confidence?: number;
}

export interface ChatSession {
  messages: ChatMessage[];
  language: LanguageType;
  documents: DocumentInfo[];
}

export interface DocumentInfo {
  id: string;
  name: string;
  type: string;
  content?: string;
  processed: boolean;
}

export interface SentimentResponse {
  sentiment: SentimentType;
  confidence: number;
  language: LanguageType;
}

@Injectable({
  providedIn: 'root'
})
export class ChatService {
  private readonly API_URL = '';

  constructor(private http: HttpClient) { }

  analyzeSentiment(text: string, language: LanguageType): Observable<SentimentResponse> {
    return this.http.post<SentimentResponse>(`${this.API_URL}/analyze-sentiment`, {
      text,
      language 
    }).pipe(
      catchError(error => {
        console.error('Sentiment analysis error:', error);
        return of({
          sentiment: 'neutral',
          confidence: 0,
          language
        } as SentimentResponse);
      })
    );
  }

  searchDocuments(query: string, language: LanguageType): Observable<string[]> {
    return this.http.post<string[]>(`${this.API_URL}/search`, {
      query,
      language 
    }).pipe(
      catchError(error => {
        console.error('Search error:', error);
        return of([]);
      })
    );
  }


  processDocument(file: File): Observable<DocumentInfo> {
    const formData = new FormData();
    formData.append('file', file);

    return this.http.post<DocumentInfo>(`${this.API_URL}/process-document`, formData).pipe(
      catchError(error => {
        console.error('Document processing error:', error);
        return of({
          id: Math.random().toString(36).substring(7),
          name: file.name,
          type: file.type,
          processed: false
        });
      })
    );
  }
}