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
    * Observar o shift da operação [DONE]
    * Observar o dia da semana + shift da operação [DONE]
    * Observar o numero da semana [DONE]
    * Observar o valor da transação e o shift [DONE]
    * Observar o valor da transação, o dia da semana e o shift [JUMPED]
    * Observar clientes com preferência de compra em turno X comprando em outro turno [JUMPED]
    * Observar clientes que geralmente adquirem em um dia da semana mas comprando em outro [JUMPED]
    * Observar tempo médio de reaquisição do produto quando n é CBK e fazer a prop [JUMPED] [DONE]
    * Observar clientes que tiveram mais de 2 operações nos ultimos 2 minutos / 5 minutos / 10 minutos [JUMPED] [DONE]
    * Observar valor médio/mediana dos ultimos X dias de aquisição pelo cliente e a distribuição na proporção que eram CBK [JUMPED] [DONE]
    * Observar a qtde de CBKs nos ultimos X dias [DONE]
    * Se é VISA ou MasterCard
    * Correlação spearman com CBK [DONE]

- Features [DONE]

* Perfil Cbk geral [DONE]
    * Dia, hora refletindo o valor [DONE]
    * Numero da semana, dia, hora refletindo o valor [DONE]

* Perfil cliente/NumeroCartao baseado em X dias [DONE]
    * Qtd de CBKs [DONE]
    * Ticket médio/mediana de CBKs [DONE]
    * Shift médio que acontece os CBKs [DONE]
    * Que dia da semana e shift mais acontece CBKs [DONE]




- Identificar a bandeira do cartão
4: VISA
5: MasterCardElo

400217******7711
400217******8714
400225******8836

Com tudo de operações [Qtd de operações nos X minutos] [DONE] [D]
Qtd de CBKs [30 dias] [DONE] [D]
Tempo medio de compra total [30 dias] [DONE] [D]
Tempo geral da ultima compra [Tudo] [DONE] [D]
Valor somado dos ultimos 3 dias [DONE]  [D]
Valor medio dos ultimos 3 dias [DONE] [D]
Qtd de operações totais [D]

Qtd de compra por semana [Sem rolling]
Qtd de compra por dia [Sem rolling]

# Features selecionadas
> mean_days_operations_customer_cbk [DONE]
> last_two_minutes_cbk_ops [DONE]
> last_days_cbk [DONE]
> last_five_minutes_cbk_ops [DONE]
> last_two_minutes_cbk_all_ops [DONE]
> last_five_minutes_cbk_all_ops [DONE]
> last_purchase_time [DONE]

> É o primeiro pagamento? [DONE]
> Tempo da ultima compra em segundos [DONE]
> Tempo da segunda ultima compra em segundos [JUMPED]
> Tempo da terceira ultima compra em segundos [JUMPED]
> O valor da ultima compra é identico a da atual? [DONE]

# Modelagem

> Checar badrate [DONE]
> Checar nulos [DONE]
> Checar ECDF das variáveis nos splits [DONE]
> Pipeline [DONE]
> Construção do modelo [DONE]
> Resultados
    PR-auc [Done]
    ROC-auc [Done]
    Distribuição scores [Done]
    Plotar a arvore [Done]
    feature_importance [Done]
    tabela dos valores dos scores [Done]
        Calcular os ganhos baseado no corte do score [Done]

> Possiveis evoluções

