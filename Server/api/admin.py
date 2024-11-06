from django.contrib import admin
from .models import UploadedDocument, ChatSession, ChatMessage


@admin.register(UploadedDocument)
class UploadedDocumentAdmin(admin.ModelAdmin):
    list_display = ('title', 'uploaded_at', 'processed', 'average_sentiment')
    list_filter = ('processed', 'language')
    search_fields = ('title', 'content')


@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ('session_id', 'created_at',
                    'last_interaction', 'current_document')
    list_filter = ('created_at', 'last_interaction')
    search_fields = ('session_id',)


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    # Updated fields in list_display and list_filter
    list_display = ('session', 'is_user', 'sentiment_score', 'timestamp')
    list_filter = ('is_user', 'timestamp')
    search_fields = ('content', 'user_sentiment', 'response_sentiment')

    # Specify timestamp ordering explicitly if needed
    ordering = ('timestamp',)
