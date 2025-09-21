from django import forms

class MultiFileInput(forms.ClearableFileInput):
    allow_multiple_selected = True

class MultiFileField(forms.FileField):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("widget", MultiFileInput(attrs={'multiple': True}))
        super().__init__(*args, **kwargs)

    def clean(self, data, initial=None):
        """Override clean() to accept multiple files"""
        if not data:
            raise forms.ValidationError("No files uploaded!")
        if not isinstance(data, (list, tuple)):
            data = [data]
        return data

class WorkerUploadForm(forms.Form):
    name = forms.CharField(max_length=100)
    images = MultiFileField(required=True)
