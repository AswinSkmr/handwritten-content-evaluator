from reportlab.pdfgen import canvas

c = canvas.Canvas("data/test_samples/sample1.pdf")
c.drawString(100, 750, "This is a test PDF document for extraction.")
c.drawString(100, 730, "It has multiple lines of real, selectable text.")
c.save()