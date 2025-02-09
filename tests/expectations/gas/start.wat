(module
  (type (;0;) (func (param i32 i32)))
  (type (;1;) (func))
  (type (;2;) (func (param i64)))
  (import "env" "ext_return" (func $ext_return (;0;) (type 0)))
  (import "env" "memory" (memory (;0;) 1 1))
  (import "env" "gas_counter" (global (;0;) (mut i64)))
  (func $start (;1;) (type 1)
    i64.const 4
    call 3
    i32.const 8
    i32.const 4
    call $ext_return
    unreachable
  )
  (func (;2;) (type 1))
  (func (;3;) (type 2) (param i64)
    global.get 0
    local.get 0
    i64.sub
    global.set 0
    global.get 0
    i64.const 0
    i64.lt_s
    if ;; label = @1
      unreachable
    end
  )
  (export "call" (func 2))
  (start $start)
  (data (;0;) (i32.const 8) "\01\02\03\04")
)