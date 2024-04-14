"""
URL configuration for uploder project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from captionimg.views import hotel_image_view,success,analyze,display_hotel_images,recent,submission,attention_den,attention_xcep,cap11,cap1,cap22,cap2,attention_result,den_gru,den_lstm
 
urlpatterns = [
    path('admin/', admin.site.urls),
    path('', hotel_image_view, name='home'),
    path('success', success, name='success'),
    path('images', display_hotel_images, name = 'images'),
    path('recent', recent, name = 'recent'),
    path('Xception_GRU',cap22,name='cap_xcepgru'),
    path('Xception_LSTM',cap2,name='cap_xceplstm'),
    path('VGG16_GRU',cap11,name='cap_vgggru'),
    path('VGG16_LSTM',cap1,name='cap_vgglstm'),
    path('Attention_GRU_VGG16',attention_result,name='cap_attenvgg'),
    path('Attention_GRU_DEN',attention_den,name='cap_attenden'),
    path('Attention_GRU_Xcep',attention_xcep,name='cap_attenxcep'),
    path('DenseNet_GRU',den_gru,name='cap_denvgg'),
    path('DenseNet_LSTM',den_lstm,name='cap_denlstm'),
    path('submission',submission,name="sub"),
    path('analyze',analyze,name="analyze"),
]
 
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,
                          document_root=settings.MEDIA_ROOT)

