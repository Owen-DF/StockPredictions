class Buy:
    def __init__(self, purchasePrice, quantity, companyName, date):
        self.purchasePrice = purchasePrice
        self.quantity = quantity
        self.companyName = companyName
        self.date = date
      

    def __str__(self):
        return f" Company name: {self.companyName}, Quantity: {self.quantity}, Purchase price: {self.purchasePrice}, Date: {self.date}"
    

    def value(self):
        return self.quantity * self.purchasePrice