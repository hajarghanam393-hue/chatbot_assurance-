import { Component, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ChatService } from '../chat';

interface Message {
  role: 'user' | 'bot';
  text: string;
}

@Component({
  selector: 'app-chat',
  standalone: true,
  imports: [FormsModule],
  templateUrl: './chat.html',
  styleUrl: './chat.css'
})
export class Chat {
  messages = signal<Message[]>([]);
  userInput = signal('');
  loading = signal(false);

  constructor(private chatService: ChatService) {}

  send() {
    const query = this.userInput().trim();
    if (!query) return;

    this.messages.update(msgs => [...msgs, { role: 'user', text: query }]);
    this.userInput.set('');
    this.loading.set(true);

    this.chatService.sendMessage(query).subscribe({
      next: (res) => {
        this.messages.update(msgs => [...msgs, { role: 'bot', text: res.answer }]);
        this.loading.set(false);
      },
      error: () => {
        this.messages.update(msgs => [...msgs, { role: 'bot', text: "Erreur : impossible de contacter le serveur." }]);
        this.loading.set(false);
      }
    });
  }
}