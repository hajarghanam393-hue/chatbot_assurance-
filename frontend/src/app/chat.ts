import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface ChatResponse {
  answer: string;
}

@Injectable({
  providedIn: 'root'
})
export class ChatService {
  private apiUrl = 'http://localhost:8000/chat';

  constructor(private http: HttpClient) {}

  sendMessage(query: string): Observable<ChatResponse> {
    return this.http.post<ChatResponse>(this.apiUrl, { query });
  }
}