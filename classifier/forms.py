from django import forms


class ContactForm(forms.Form):
    Text = forms.CharField(widget= forms.Textarea)

