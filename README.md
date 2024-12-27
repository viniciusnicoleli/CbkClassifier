# CbkClassifier
Identificando transações que possam se tornar chargeback.

> EDA
- O que caracteriza as transações desse cliente? [DONE]
    * Quanto os clientes desse e-commerce gastam em um mes? [DONE]
        * Quanto realmente é faturamento [DONE]
        * Quanto realmente é cancelado com chargeback [DONE]
        * Quantos % de perda de venda? [DONE]
        
    * Em que momento do dia [DONE]
        * Influencia na qtde de compras? [DONE]
        * Influencia no valor gasto por cada compra? [DONE]
        * O Valor vendido é maior? [DONE]

    * O dia da semana [DONE]
        * Influencia na qtde de compras? [DONE]
        * Influencia no valor gasto por cada compra? [DONE]
        * O Valor vendido é maior? [DONE]

        # Heatmap [Dia da semana | Shift | Valor e qtd vendida] [DONE]
        # Dia da semana influencia a qtd vendida [Boxplot] [DONE]
        # Dia da semana influencia o valor de venda [Ridgeline] [Boxplot] [DONE]

    * O numero da semana [DONE]
        * Influencia na qtde de compras? [DONE]
        * Influencia no valor gasto por cada compra? [DONE]
        * O Valor vendido é maior? [DONE]
    
    * Clientes [Mês completo] [JUMPED]
        * Classificar clientes compradores
            * Proporção de clientes que compram muito
            * Proporção de clientes que compram medianamente
            * Proporção de clientes que compram pouco
        
        * Estatísticas
            * Classe Clientes
                * Em que momento do dia eles compram mais
                * Qual valor gasto por semana
                * De quanto em quanto tempo eles compram
    
    * Estatísticas gerais [DONE]
        * Quantidade de operações diárias [DONE]
        * Quantidade de revenue diário [DONE]

- Qual o perfil das transações que retornam chargeback?
    # Observar olhando CBK 1/0
    * Observar o shift da operação
    * Observar o dia da semana + shift da operação
    * Observar o numero da semana
    * Observar o valor da transação e o shift
    * Observar o valor da transação, o dia da semana e o shift

    # Clientes preferencialmente noturnos ou etc

- Features

* Perfil Cbk geral
    * Dia, hora refletindo o valor
    * Numero da semana, dia, hora refletindo o valor

* Perfil cliente/NumeroCartao baseado em X dias
    * Qtd de CBKs
    * Ticket médio/mediana de CBKs
    * Shift médio que acontece os CBKs
    * Que dia da semana e shift mais acontece CBKs




- Identificar a bandeira do cartão
4: VISA
5: MasterCardElo