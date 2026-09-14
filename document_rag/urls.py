from django.urls import path, include
from backend.views import serve_spa

urlpatterns = [
    path('', serve_spa, name='spa_index'),
    path('api/', include('backend.urls')),
]
