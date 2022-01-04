### A Pluto.jl notebook ###
# v0.17.4

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 31f52a36-948a-4441-aeaf-1ca661e3ee63
using PlutoUI

# ╔═╡ 63eb5b58-0cc7-4a69-8f71-6a76e8831c23
using LinearAlgebra: norm

# ╔═╡ 6315031b-3038-4b17-acee-293c01d16231
md"""
# Solve-for-𝑢

**Find feasible shift-rule with given support**

Given a set set of frequencies $\Xi_+$ and a set $A$, this notebook demonstrates how to solve the system of equations to obtain a feasible shift rule whose support is contained in $A$.  If none exists, the “closest to feasible” (in L²-norm) shift rule with that support is computed.

###### Author & Copyright
This Pluto notebook is part of the supplementary material for the paper “Optimizal parameter shift rules for variational quantum circuits”.

Dirk Oliver Theis                              \
Assoc. Prof. Theoretical Computer Science      \
University of Tartu, Estonia
"""

# ╔═╡ d9ee4c7e-07a2-4160-8614-038bd89371cb
begin
	import Base.:*

	const ⋅ = Base.:*

	const ℝ = BigFloat
	const ℤ = Int64
	const ℚ = Rational{ℤ}

	const ℂ  = Complex{ℝ}
	const ℜ  = real
	const ℑ  = imag
	;                      setprecision(ℝ,65536)
	const 𝒊  = ℂ(im)
	const π𝒊 = ℂ(π⋅im)
	const 𝒊π = π𝒊
	;                      setprecision(ℝ,256)
	md"""We use definitions for nicer math notation, e.g., "⋅" instead of "$*$". The type "ℝ" is floating point with the chosen precision."""
end

# ╔═╡ 5e1eb561-3f5b-45e1-91e8-76b3ad8bbb9b
md"""
###### The functions for solving
"""

# ╔═╡ 1b768f46-1446-4cf8-8285-fc46817c966c
begin
	function make_SLE(; Ξ₊, α, A)
		L   = length(Ξ₊)
		S   = length(A)
		E   = Matrix{ℝ}(undef,  1+2L, S )
		rhs = Vector{ℝ}(undef,  1+2L)
		# 1st row: Frequency 0
		for s = 1:S
			E[1,s] = 1
		end
		rhs[1] = 0
		# cosine / real-part rows:
		for ℓ = 1:L
			for s = 1:S
				E[1+   ℓ , s] =  cos(Ξ₊[ℓ]⋅A[s]⋅2π)
			end
			rhs[  1+   ℓ]     = ℜ(𝒊^α)⋅(Ξ₊[ℓ]⋅2π)^α
		end
		# sine / imaginary-part rows:
		for ℓ = 1:L
			for s = 1:S
				E[1+L+ ℓ , s] = -sin(Ξ₊[ℓ]⋅A[s]⋅2π)
			end
			rhs[  1+L+ ℓ]     = ℑ(𝒊^α)⋅(Ξ₊[ℓ]⋅2π)^α
		end
		return (E=E,rhs=rhs)
	end
	md"Function `make_SLE(; Ξ₊, α, A) :: NamedTuple{E,rhs}` creates matrix and RHS vector for the system of linear equations:"
end

# ╔═╡ 5f115289-d371-4dbb-9eb2-eac5546abbb0
begin
	function solve_it(; Ξ₊, α, A)
		LSE = make_SLE(; Ξ₊, α, A)
		E   = LSE.E
		rhs = LSE.rhs

		u      = E\rhs
		error  = norm(rhs - E⋅u , Inf)

		return (u=u,error=error)
	end
	md"Function `solve_it(; Ξ₊, α, A) :: @NamedTuple{u,error}` does the actual work:"
end

# ╔═╡ 78fe3596-a1cc-4490-b8be-9e3149c0f50c
md"###### Run it!"

# ╔═╡ a747edfd-82f1-4fb0-b128-8182e821edce
md"Define your data:"

# ╔═╡ 86b78588-eaa6-4784-8226-d6931c83196a
α = 1

# ╔═╡ 1ba3dfb7-86cb-4a08-a803-c287c4255e8e
if !( 1 ≤ α ≤ 4 )
	Markdown.MD(Markdown.Admonition("danger", "Out of range", [
		md"""
		Please choose $\alpha \in \{1,\dots,4\}$.
		"""]))
else
	md"Range check for $\alpha$: ✅"
end

# ╔═╡ bfeb59e9-eaf3-4993-9639-9f61d5347feb
Ξ₊ = ℚ[ 2, 3 ]

# ╔═╡ 0e0d14c5-050d-4521-8d14-73e10f15476d
A = ℚ[ -1//12,  1//12,  5//12 #=, -5//12=# ]

# ╔═╡ a547012b-8d63-4204-9ac0-5a7cf4ea7c7d
md"""
|  |  |
|:--:|--:|
| α     | $α             |
| Ξ₊    | $(length(Ξ₊))  |
| \|A\| | $(length(A))   |
"""

# ╔═╡ f3533f9f-d76d-4d6a-904a-c2bd6e63acf9
md"""
###### Choose the floating point precision: $(@bind PRECISION Slider(64:64:16384 ; default=1024))
Play with the precision to see whether there really exists a feasible shift rule with the given support: If there is one, then the error should tend towards 0 when the precision is increased.
"""

# ╔═╡ 6d9be377-974c-42a9-b902-5cd6a38fc563
md"""
Setting the precision of floating-point arithmetic to $( setprecision(ℝ,PRECISION) ).
\
(Precision of 64-bit C-style `double` is 64.)
"""

# ╔═╡ 7ca6efd2-237d-4705-99b9-7fb9dc7a180b
begin
	PRECISION  # re-run this cell when precision changes

	ref_u = Ref{Vector{ℝ}}(ℝ[])

	with_terminal() do
		mylog(x) = Int(floor(log10(x))) 
		sgn(x)   = x≥0 ? "+" : "-"
		errwsgn(x) = x≠0 ? "$(sgn(x))1e$(mylog(x))" : " 0"

		result = solve_it(; α,Ξ₊,A)

		logerror = mylog(result.error)

		Δobj     = abs( norm(result.u,1) - (2π⋅maximum(Ξ₊))^α ) # cost - opt

		println("error      ≈  1e$(logerror)     at precision $(precision(ℝ))")
		println("cost - opt ≈ $(errwsgn(Δobj))")
		ref_u[]     = result.u
		nothing ;
	end
end	

# ╔═╡ eff0ef1f-5c13-46cb-b771-6351dd14af85
md"###### Result:"

# ╔═╡ d668c890-73fc-426a-b2a1-d1e4e93f6540
u = ref_u[]

# ╔═╡ 1d8927e2-1d33-4785-a70d-9fb70ccb3b76
md"""
!!! license "MIT License"
	Copyright 2021 Dirk Oliver Theis

	Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

	The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
PlutoUI = "~0.7.22"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "abb72771fd8895a7ebd83d5632dc4b989b022b5b"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.2"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[HypertextLiteral]]
git-tree-sha1 = "2b078b5a615c6c0396c77810d92ee8c6f470d238"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.3"

[[IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "ae4bbcadb2906ccc085cf52ac286dc1377dceccc"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.1.2"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "Dates", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "UUIDs"]
git-tree-sha1 = "565564f615ba8c4e4f40f5d29784aa50a8f7bbaf"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.22"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╟─6315031b-3038-4b17-acee-293c01d16231
# ╠═31f52a36-948a-4441-aeaf-1ca661e3ee63
# ╟─d9ee4c7e-07a2-4160-8614-038bd89371cb
# ╟─5e1eb561-3f5b-45e1-91e8-76b3ad8bbb9b
# ╠═1b768f46-1446-4cf8-8285-fc46817c966c
# ╠═63eb5b58-0cc7-4a69-8f71-6a76e8831c23
# ╠═5f115289-d371-4dbb-9eb2-eac5546abbb0
# ╟─78fe3596-a1cc-4490-b8be-9e3149c0f50c
# ╟─a747edfd-82f1-4fb0-b128-8182e821edce
# ╠═86b78588-eaa6-4784-8226-d6931c83196a
# ╟─1ba3dfb7-86cb-4a08-a803-c287c4255e8e
# ╠═bfeb59e9-eaf3-4993-9639-9f61d5347feb
# ╠═0e0d14c5-050d-4521-8d14-73e10f15476d
# ╟─a547012b-8d63-4204-9ac0-5a7cf4ea7c7d
# ╟─f3533f9f-d76d-4d6a-904a-c2bd6e63acf9
# ╟─6d9be377-974c-42a9-b902-5cd6a38fc563
# ╠═7ca6efd2-237d-4705-99b9-7fb9dc7a180b
# ╟─eff0ef1f-5c13-46cb-b771-6351dd14af85
# ╟─d668c890-73fc-426a-b2a1-d1e4e93f6540
# ╟─1d8927e2-1d33-4785-a70d-9fb70ccb3b76
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
