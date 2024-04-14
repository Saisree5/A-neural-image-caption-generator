from django import forms
from .models import Img_model
class Img_form(forms.ModelForm):
    class Meta:
        model=Img_model
        fields=['img']