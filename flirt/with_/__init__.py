# easter egg package to make flirting fun :-D

from ..simple.empatica import get_features_for_empatica_archive as empatica

__all__ = [
    "empatica",
    "me",
]


# Want FLIRT to flirt with you? ;-)
# import flirt.with_
# print(flirt.with_.me())
def me():
    fwm = ("WW91J3JlIHN3ZWV0ZXIgdGhhbiAzLjE0IQpZb3UgYXJlIG9uZSB3ZWxsLWRlZmluZWQgZnVuY3Rp"
           "b24hCk15IGxvdmUgZm9yIHlvdSBnb2VzIG9uIGxpa2UgdGhlIHZhbHVlIG9mIHBpLgpJJ20gbm90"
           "IGJlaW5nIG9idHVzZSwgYnV0IHlvdSdyZSBzbyBhY3V0ZSEKSSB3YW50IG91ciBsb3ZlIHRvIGJl"
           "IGxpa2UgcGksIGlycmF0aW9uYWwgYW5kIG5ldmVyIGVuZGluZy4KSSBhbSBjb3NpbmUgc3F1YXJl"
           "ZCBhbmQgeW91IGFyZSBzaW5lIHNxdWFyZWQuIFRvZ2V0aGVyLCB3ZSBhcmUgb25lLgpBcmUgeW91"
           "IHRoZSBzcXVhcmUgcm9vdCBvZiAtMT8gQmVjYXVzZSB5b3UgY2FuJ3QgYmUgcmVhbCEKSSBuZWVk"
           "IHNvbWUgYW5zd2VycyBmb3IgbXkgbWF0aCBob21ld29yay4gUXVpY2suIFdoYXQncyB5b3VyIG51"
           "bWJlcj8KWW91IG11c3QgYmUgYSA5MC1kZWdyZWUgYW5nbGUsIGJlY2F1c2UgeW91J3JlIGxvb2tp"
           "bmcgYWxsIHJpZ2h0IQpJZiB5b3Ugd2VyZSBhbiBhbmdsZSwgeW914oCZZCBiZSBhY3V0ZSBvbmUu"
           "CllvdSBhbmQgSSBhZGQgdXAgYmV0dGVyIHRoYW4gdGhlIFJpZW1hbm4gc3VtLgpDYW4gSSBoYXZl"
           "IHlvdXIgc2lnbmlmaWNhbnQgZGlnaXRzPwpZb3UgbXVzdCBiZSB0aGUgc3F1YXJlIHJvb3Qgb2Yg"
           "MiBiZWNhdXNlIEkgZmVlbCBpcnJhdGlvbmFsIGFyb3VuZCB5b3UuCkRvIHlvdSB3YW50IHRvIHNo"
           "YXJlIHNvbWUgZWxlY3Ryb25zPyBNYXliZSB3ZSBjb3VsZCBoYXZlIGEgc3RhYmxlIHJlbGF0aW9u"
           "c2hpcC4=")
    import base64
    import random
    return random.choice(str(base64.b64decode(fwm), "utf-8").split('\n'))
