from mm.avellaneda import ASParams, AvellanedaStoikovStrategy

params = ASParams(
    gamma=0.1,
    sigma=1.0,
    k=1.5,
    A=50,
    T=100,
    dt=1.0,
)

strat = AvellanedaStoikovStrategy(params)

S = 100.0
q_values = [-5, -2, 0, 2, 5]

print("Inventory → Reservation price → Bid/Ask")
for q in q_values:
    bid, ask, _, _ = strat.compute_quotes(S=S, q=q, t=0)
    print(f"{q:3} → bid={bid:.3f}, ask={ask:.3f}")
