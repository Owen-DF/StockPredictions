class Buy:
    def __init__(self, purchasePrice, quantity, companyName):
        self.purchasePrice = purchasePrice
        self.quantity = quantity
        self.companyName = companyName
      

    def __str__(self):
        return f" Company name: {self.companyName}, Quantity: {self.quantity}, Purchase price: {self.purchasePrice}"
    

    def value(self):
        return self.quantity * self.purchasePrice