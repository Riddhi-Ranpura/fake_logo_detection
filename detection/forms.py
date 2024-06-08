from django import forms

class LogoUploadForm(forms.Form):
    logo = forms.ImageField()
#