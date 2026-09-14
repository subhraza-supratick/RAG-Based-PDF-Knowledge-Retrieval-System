from django.urls import path
from backend import views

urlpatterns = [
    path('health/', views.health_check, name='health_check'),
    path('documents/upload/', views.upload_document, name='upload_document'),
    path('documents/<int:document_id>/index/', views.index_document, name='index_document'),
    path('documents/', views.list_documents, name='list_documents'),
    path('documents/<int:document_id>/', views.remove_document, name='remove_document'),
    path('ask/', views.ask_question, name='ask_question'),
]
