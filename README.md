~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Purchases-City-Council.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This dataset has a historical collection of purchase card transactions for the Birmingham City Council.
The time window is from April 2014 to February 2018.
The method used for the forcasting analysis is **ARIMA Model**.

Clic [here](https://data.birmingham.gov.uk/dataset/purchase-card-transactions) to see the dataset website.

------------------------------------------------------------------------------------
- TO SEE THE FULL CODE ON JUPYTER NOTEBOOKS CHOOSE THE FILE: Purchases.ipynb       -
- TO SEE THE FULL CODE ON PYTHON IDE CHOOSE THE FILE: Purchases_Spyder.py          -
------------------------------------------------------------------------------------

-----------------------
Variables description
-----------------------
1. **TRANS DATE:** date of transaction
2. **TRANS VAT DESC:** VAT transaction description
    - The VAT tax (Value-added tax) is payable by any taxable person making a taxable supply 
      (‘the supplier’) of goods or services, unless it is payable by another person.
    - The VAT rates: ['VR':reduced rate,'VZ':Zero rate,'VL':Leisure, 'VT':Transport,'VE':Education,'VS':Standard rate]
     
3. **ORIGINAL GROSS AMT:** original gross amount
     - The income amount calculated for tax purposes to be paid.
     
4. **MERCHANT NAME:** A qué proveedor se le compró.
5. CARD NUMBER
6. TRANS CAC CODE 1: Client Account Credit Transaction code
7. **TRANS CAC DESC 1:** Concepto del gasto (viáticos,gasolina,etc)
8. TRANS CAC CODE 2
9. **TRANS CAC DESC 2:** Qué área de la municipalidad compró.
10. TRANS CAC CODE 3
11. **DIRECTORATE:** Bajo qué gerencia o directorio se encuentra el área.
 
Birmingham City Council Quote (Spanish)
-------------------------------------------------------------------------------------------------------------------
*En virtud del Código de prácticas recomendadas para las autoridades locales sobre transparencia de datos,        -
se alienta a los ayuntamientos a publicar todas las transacciones con tarjetas de compra corporativas.            -
Ya publicamos detalles de todos nuestros gastos relevantes de más de £ 500 en nuestra página Pagos a proveedores, -
y continuaremos haciéndolo.*                                                                                      -
-------------------------------------------------------------------------------------------------------------------
